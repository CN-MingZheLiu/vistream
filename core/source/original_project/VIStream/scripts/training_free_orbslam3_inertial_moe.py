#!/usr/bin/env python3
"""Audited zero-parameter gate for ORB-SLAM3 inertial trajectory experts."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256(resolved)}


def verify_artifact(value: dict[str, str], label: str) -> None:
    path = Path(value["path"])
    if sha256(path) != value["sha256"]:
        raise ValueError(f"sealed artifact changed: {label}: {path}")


def poses_4x4(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1:] == (4, 4):
        return values.copy()
    if values.shape[1:] != (3, 4):
        raise ValueError(f"expected [N,3,4] or [N,4,4], got {values.shape}")
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


def timestamp_ns_from_stem(stem: str) -> int:
    if "." in stem:
        return int((Decimal(stem) * Decimal(1_000_000_000)).to_integral_exact())
    return int(stem)


def cache_timestamps_ns(
    archive: np.lib.npyio.NpzFile, expected_count: int | None = None
) -> np.ndarray:
    if "image_paths" in archive.files:
        image_timestamps = np.asarray(
            [timestamp_ns_from_stem(Path(str(path)).stem) for path in archive["image_paths"]],
            dtype=np.int64,
        )
        if expected_count is None or len(image_timestamps) == expected_count:
            return image_timestamps
        if "metric_indices" in archive.files:
            indices = np.asarray(archive["metric_indices"], dtype=np.int64)
            if len(indices) == expected_count:
                return image_timestamps[indices]
    if "timestamps_ns" in archive.files:
        return np.asarray(archive["timestamps_ns"], dtype=np.int64)
    if "timestamps" in archive.files:
        return np.rint(np.asarray(archive["timestamps"], dtype=np.float64) * 1e9).astype(
            np.int64
        )
    raise KeyError("cache has no image_paths, timestamps_ns, or timestamps")


def load_integer_timestamps(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text().splitlines():
        token = line.strip().split(" ", 1)[0]
        if token and not token.startswith("#"):
            values.append(int(token))
    timestamps = np.asarray(values, dtype=np.int64)
    if len(timestamps) == 0 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("input timestamps are empty or not strictly increasing")
    return timestamps


def parse_orb_euroc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if rows.ndim != 2 or rows.shape[1] != 8 or len(rows) == 0:
        raise ValueError(f"expected nonempty ORB EuRoC trajectory [N,8], got {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError("ORB trajectory contains non-finite values")
    timestamps_ns = np.rint(rows[:, 0]).astype(np.int64)
    quaternion_norms = np.linalg.norm(rows[:, 4:8], axis=1)
    if np.any(quaternion_norms <= 1e-12):
        raise ValueError("ORB trajectory contains a zero quaternion")
    poses = np.repeat(np.eye(4)[None], len(rows), axis=0)
    poses[:, :3, 3] = rows[:, 1:4]
    poses[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
    return timestamps_ns, poses


def match_timestamps_ns(
    query: np.ndarray, reference: np.ndarray, tolerance_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    if tolerance_ns < 0:
        raise ValueError("timestamp tolerance must be nonnegative")
    right = np.searchsorted(reference, query)
    right = np.clip(right, 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    choose_left = np.abs(reference[left] - query) <= np.abs(reference[right] - query)
    indices = np.where(choose_left, left, right).astype(np.int64)
    errors = np.abs(reference[indices] - query)
    if np.any(errors > tolerance_ns):
        worst = int(np.argmax(errors))
        raise ValueError(
            f"timestamp mismatch {int(query[worst])} versus "
            f"{int(reference[indices[worst]])}, error {int(errors[worst])} ns"
        )
    if len(np.unique(indices)) != len(indices):
        raise ValueError("multiple timestamps map to one input frame")
    return indices, errors


def final_map_count(text: str) -> int:
    matches = re.findall(r"There are\s+(\d+)\s+maps in the atlas", text)
    if matches:
        return int(matches[-1])
    return text.count("Creation of new map with id")


def run_features(run_dir: Path, all_timestamps_ns: np.ndarray, tolerance_ns: int) -> dict:
    trajectories = sorted(run_dir.glob("f_*.txt"))
    keyframes = sorted(run_dir.glob("kf_*.txt"))
    log_path = run_dir / "run.log"
    if len(trajectories) != 1 or len(keyframes) != 1 or not log_path.is_file():
        raise ValueError(f"run directory lacks one f_*, one kf_*, and run.log: {run_dir}")
    text = log_path.read_text(errors="replace")
    raw_timestamps_ns, poses = parse_orb_euroc(trajectories[0])
    nonmonotonic = int(np.sum(np.diff(raw_timestamps_ns) <= 0))
    matched, errors = match_timestamps_ns(
        raw_timestamps_ns, all_timestamps_ns, tolerance_ns
    )
    span = int(matched[-1] - matched[0] + 1)
    internal_missing = max(0, span - len(matched))
    features = {
        "run_dir": str(run_dir.resolve()),
        "output_frames": len(matched),
        "input_frames": len(all_timestamps_ns),
        "tracking_completeness": float(len(matched) / len(all_timestamps_ns)),
        "first_tracked_frame": int(matched[0]),
        "last_tracked_frame": int(matched[-1]),
        "missing_prefix_frames": int(matched[0]),
        "missing_suffix_frames": int(len(all_timestamps_ns) - 1 - matched[-1]),
        "internal_missing_frames": internal_missing,
        "internal_missing_fraction": float(internal_missing / max(span, 1)),
        "timestamp_nonmonotonic_count": nonmonotonic,
        "maximum_timestamp_error_ns": int(errors.max(initial=0)),
        "final_map_count": final_map_count(text),
        "active_map_reset_count": text.count("SYSTEM-> Reseting active map"),
        "viba1_complete": "end VIBA 1" in text,
        "viba2_complete": "end VIBA 2" in text,
        "shutdown_complete": "Shutdown" in text,
        "failed_local_map_count": text.count("Fail to track local map!"),
        "loop_detection_count": text.count("*Loop detected"),
        "keyframes": sum(1 for line in keyframes[0].read_text().splitlines() if line.strip()),
    }
    return {
        "features": features,
        "timestamps_ns": raw_timestamps_ns,
        "matched_frame_indices": matched,
        "poses": poses,
        "artifacts": {
            "trajectory": artifact(trajectories[0]),
            "keyframes": artifact(keyframes[0]),
            "run_log": artifact(log_path),
        },
    }


def frozen_health_gate(features: dict, policy: dict) -> dict:
    valid = bool(
        features["viba2_complete"]
        and features["shutdown_complete"]
        and features["final_map_count"] == 1
        and features["timestamp_nonmonotonic_count"] == 0
        and features["tracking_completeness"]
        >= float(policy["minimum_tracking_completeness"])
    )
    completeness = float(features["tracking_completeness"])
    internal = float(features["internal_missing_fraction"])
    if not valid:
        environment = "vio_invalid_visual_fallback"
        weight = 0.0
    elif (
        completeness >= float(policy["trusted_tracking_completeness"])
        and internal <= float(policy["maximum_trusted_internal_missing_fraction"])
    ):
        environment = "stable_metric_vio"
        weight = float(policy["trusted_orb_weight"])
    elif completeness >= float(policy["advisory_tracking_completeness"]):
        environment = "partially_observed_metric_vio"
        weight = float(policy["advisory_orb_weight"])
    else:
        environment = "weakly_observed_metric_vio"
        weight = float(policy["weak_orb_weight"])
    return {
        "network": "frozen_vio_health_gate_v1",
        "trainable_parameters": 0,
        "valid_orb_inertial_expert": valid,
        "environment": environment,
        "orb_weight": weight,
        "visual_weight": 1.0 - weight,
    }


def replica_rank(record: dict, policy: dict) -> tuple:
    features = record["features"]
    gate = frozen_health_gate(features, policy)
    return (
        int(gate["valid_orb_inertial_expert"]),
        float(features["tracking_completeness"]),
        -float(features["internal_missing_fraction"]),
        -int(features["active_map_reset_count"]),
        int(features["keyframes"]),
    )


def boundary_alignment(
    visual: np.ndarray,
    cache_indices: np.ndarray,
    overlap_orb: np.ndarray,
    side: str,
) -> tuple[object, dict[str, float | int]]:
    from vistream.metrics import umeyama_alignment

    if side not in {"prefix", "suffix"}:
        raise ValueError("boundary side must be prefix or suffix")
    candidates = [value for value in (16, 32, 64, 128, 256) if value <= len(cache_indices)]
    if not candidates:
        candidates = [len(cache_indices)]
    records = []
    for window in candidates:
        if side == "prefix":
            fit_cache = cache_indices[:window]
            fit_orb = overlap_orb[:window]
            expected_step = overlap_orb[1, :3, 3] - overlap_orb[0, :3, 3]
            boundary_visual_index = int(cache_indices[0] - 1)
            boundary_orb_position = overlap_orb[0, :3, 3]
            hold_count = min(window, len(cache_indices) - window)
            hold_cache = cache_indices[window : window + hold_count]
            hold_orb = overlap_orb[window : window + hold_count]
            scale_steps = np.diff(overlap_orb[: min(32, len(overlap_orb)), :3, 3], axis=0)
        else:
            fit_cache = cache_indices[-window:]
            fit_orb = overlap_orb[-window:]
            expected_step = overlap_orb[-1, :3, 3] - overlap_orb[-2, :3, 3]
            boundary_visual_index = int(cache_indices[-1] + 1)
            boundary_orb_position = overlap_orb[-1, :3, 3]
            hold_count = min(window, len(cache_indices) - window)
            hold_cache = cache_indices[-window - hold_count : -window]
            hold_orb = overlap_orb[-window - hold_count : -window]
            scale_steps = np.diff(overlap_orb[-min(32, len(overlap_orb)) :, :3, 3], axis=0)
        alignment = umeyama_alignment(
            visual[fit_cache, :3, 3], fit_orb[:, :3, 3], with_scale=True
        )
        aligned_positions = alignment.apply(visual[:, :3, 3])
        if side == "prefix":
            join_step = boundary_orb_position - aligned_positions[boundary_visual_index]
        else:
            join_step = aligned_positions[boundary_visual_index] - boundary_orb_position
        median_step = float(np.median(np.linalg.norm(scale_steps, axis=1)))
        magnitude_error = abs(float(np.linalg.norm(join_step)) - median_step) / max(
            median_step, 1e-9
        )
        direction_error = 1.0 - float(np.dot(join_step, expected_step)) / max(
            float(np.linalg.norm(join_step) * np.linalg.norm(expected_step)), 1e-9
        )
        fit_error = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (aligned_positions[fit_cache] - fit_orb[:, :3, 3]) ** 2,
                        axis=1,
                    )
                )
            )
            / max(median_step * window, 1e-9)
        )
        holdout_error = 0.0
        if hold_count:
            holdout_error = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (aligned_positions[hold_cache] - hold_orb[:, :3, 3]) ** 2,
                            axis=1,
                        )
                    )
                )
                / max(median_step * hold_count, 1e-9)
            )
        score = magnitude_error + direction_error + fit_error + holdout_error
        records.append(
            (
                score,
                alignment,
                {
                    "window_frames": window,
                    "score": score,
                    "boundary_jump": float(np.linalg.norm(join_step)),
                    "median_orb_step": median_step,
                    "fit_error": fit_error,
                    "holdout_error": holdout_error,
                    "scale": float(alignment.scale),
                },
            )
        )
    _, alignment, diagnostics = min(records, key=lambda value: (value[0], value[2]["window_frames"]))
    return alignment, diagnostics


def align_and_blend(
    visual: np.ndarray,
    cache_to_input: np.ndarray,
    orb_input_frames: np.ndarray,
    orb_poses: np.ndarray,
    orb_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    from vistream.metrics import umeyama_alignment

    visual = poses_4x4(visual)
    input_to_orb = {int(frame): index for index, frame in enumerate(orb_input_frames)}
    cache_indices = np.asarray(
        [index for index, frame in enumerate(cache_to_input) if int(frame) in input_to_orb],
        dtype=np.int64,
    )
    if orb_weight == 0.0:
        mask = np.zeros(len(visual), dtype=bool)
        mask[cache_indices] = True
        return visual.copy(), mask, {
            "scale": 1.0,
            "rotation_angle_rad": 0.0,
            "overlap_frames": len(cache_indices),
            "cache_tracking_completeness": float(len(cache_indices) / len(visual)),
            "prefix_bridge": None,
            "suffix_bridge": None,
        }
    if len(cache_indices) < 3:
        raise ValueError("fewer than three cache frames overlap the ORB trajectory")
    orb_indices = np.asarray(
        [input_to_orb[int(cache_to_input[index])] for index in cache_indices],
        dtype=np.int64,
    )
    overlap_orb = orb_poses[orb_indices]
    alignment = umeyama_alignment(
        visual[cache_indices, :3, 3], overlap_orb[:, :3, 3], with_scale=True
    )
    aligned_visual = visual.copy()
    aligned_visual[:, :3, 3] = alignment.apply(visual[:, :3, 3])
    aligned_visual[:, :3, :3] = alignment.rotation @ visual[:, :3, :3]
    candidate = aligned_visual.copy()
    prefix_bridge = None
    suffix_bridge = None
    if cache_indices[0] > 0 and len(cache_indices) >= 2:
        prefix_alignment, prefix_bridge = boundary_alignment(
            visual, cache_indices, overlap_orb, "prefix"
        )
        candidate[: cache_indices[0], :3, 3] = prefix_alignment.apply(
            visual[: cache_indices[0], :3, 3]
        )
        candidate[: cache_indices[0], :3, :3] = (
            prefix_alignment.rotation @ visual[: cache_indices[0], :3, :3]
        )
    if cache_indices[-1] < len(visual) - 1 and len(cache_indices) >= 2:
        suffix_alignment, suffix_bridge = boundary_alignment(
            visual, cache_indices, overlap_orb, "suffix"
        )
        suffix = slice(cache_indices[-1] + 1, None)
        candidate[suffix, :3, 3] = suffix_alignment.apply(visual[suffix, :3, 3])
        candidate[suffix, :3, :3] = suffix_alignment.rotation @ visual[suffix, :3, :3]
    if orb_weight > 0.0:
        candidate[cache_indices, :3, 3] = (
            (1.0 - orb_weight) * aligned_visual[cache_indices, :3, 3]
            + orb_weight * overlap_orb[:, :3, 3]
        )
        relative = (
            aligned_visual[cache_indices, :3, :3].transpose(0, 2, 1)
            @ overlap_orb[:, :3, :3]
        )
        candidate[cache_indices, :3, :3] = (
            aligned_visual[cache_indices, :3, :3]
            @ Rotation.from_rotvec(
                orb_weight * Rotation.from_matrix(relative).as_rotvec()
            ).as_matrix()
        )
    mask = np.zeros(len(visual), dtype=bool)
    mask[cache_indices] = True
    return candidate, mask, {
        "scale": float(alignment.scale),
        "rotation_angle_rad": float(Rotation.from_matrix(alignment.rotation).magnitude()),
        "overlap_frames": len(cache_indices),
        "cache_tracking_completeness": float(len(cache_indices) / len(visual)),
        "prefix_bridge": prefix_bridge,
        "suffix_bridge": suffix_bridge,
    }


def parse_named_artifacts(values: list[str]) -> dict[str, dict[str, str]]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"sealed artifact must be NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"invalid or duplicate sealed artifact name: {name}")
        output[name] = artifact(Path(raw_path))
    return output


def load_visual(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    # NPZ members are lazy; no ground-truth member is requested in generation.
    with np.load(args.cache, allow_pickle=False) as archive:
        visual = poses_4x4(archive[args.visual_pose_key])
        cache_ns = cache_timestamps_ns(archive, len(visual))
    if args.visual_convention == "w2c":
        visual = np.linalg.inv(visual)
    if len(visual) != len(cache_ns) or np.any(np.diff(cache_ns) <= 0):
        raise ValueError("cache visual poses and timestamps are inconsistent")
    return visual, cache_ns


def generate(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = json.loads(args.policy.read_text())
    all_ns = load_integer_timestamps(args.timestamps)
    visual, cache_ns = load_visual(args)
    cache_to_input, cache_errors = match_timestamps_ns(
        cache_ns, all_ns, args.timestamp_tolerance_ns
    )
    replicas = [
        run_features(run_dir, all_ns, args.timestamp_tolerance_ns)
        for run_dir in args.run_dir
    ]
    selected_index = max(range(len(replicas)), key=lambda index: replica_rank(replicas[index], policy))
    selected = replicas[selected_index]
    selected_features = dict(selected["features"])
    input_to_orb = set(map(int, selected["matched_frame_indices"]))
    selected_features["cache_tracking_completeness"] = float(
        sum(int(frame) in input_to_orb for frame in cache_to_input) / len(cache_to_input)
    )
    gate_features = dict(selected_features)
    gate_features["tracking_completeness"] = selected_features[
        "cache_tracking_completeness"
    ]
    gate = frozen_health_gate(gate_features, policy)
    candidate, orb_mask, alignment = align_and_blend(
        visual,
        cache_to_input,
        selected["matched_frame_indices"],
        selected["poses"],
        float(gate["orb_weight"]),
    )
    diagnostics = {
        "dataset": args.dataset,
        "sequence": args.sequence,
        "gate": gate,
        "selected_replica_index": selected_index,
        "selected_features": selected_features,
        "replicas": [record["features"] for record in replicas],
        "visual_to_orb_sim3": alignment,
        "maximum_cache_timestamp_error_ns": int(cache_errors.max(initial=0)),
    }
    candidate_path = args.output_dir / "candidate_before_gt.npz"
    np.savez_compressed(
        candidate_path,
        c2w=candidate,
        timestamps_ns=cache_ns,
        input_frame_indices=cache_to_input,
        orb_mask=orb_mask,
        orb_weight=np.asarray(gate["orb_weight"]),
    )
    diagnostics_path = args.output_dir / "inertial_gate_before_gt.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    manifest = {
        "protocol": args.protocol,
        "dataset": args.dataset,
        "sequence": args.sequence,
        "visual_pose_key": args.visual_pose_key,
        "visual_convention": args.visual_convention,
        "timestamp_tolerance_ns": args.timestamp_tolerance_ns,
        "policy": policy,
        "selected_replica_index": selected_index,
        "artifacts": {
            "cache": artifact(args.cache),
            "timestamps": artifact(args.timestamps),
            "policy": artifact(args.policy),
            "candidate": artifact(candidate_path),
            "diagnostics": artifact(diagnostics_path),
            **parse_named_artifacts(args.sealed_artifact),
        },
        "replicas": [record["artifacts"] for record in replicas],
    }
    manifest_path = args.output_dir / "manifest_before_gt.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))
    print(f"manifest_sha256={sha256(manifest_path)}")


def verify_manifest(manifest: dict) -> None:
    for name, value in manifest["artifacts"].items():
        verify_artifact(value, name)
    for index, replica in enumerate(manifest["replicas"]):
        for name, value in replica.items():
            verify_artifact(value, f"replica_{index}_{name}")


def metric(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from vistream.metrics import ate_rmse

    valid = np.isfinite(predicted).all(axis=1) & np.isfinite(target).all(axis=1)
    result = ate_rmse(predicted[valid], target[valid], "sim3")
    return {
        "frames": int(np.sum(valid)),
        "ate_rmse_m": float(result["ate_rmse"]),
        "ate_mean_m": float(result["ate_mean"]),
    }


def external_body_ground_truth(
    csv_path: Path,
    sensor_yaml: Path,
    timestamps_ns: np.ndarray,
) -> np.ndarray:
    raw = np.loadtxt(csv_path, delimiter=",", comments="#", dtype=np.float64)
    gt_ns = raw[:, 0].astype(np.int64)
    valid = (timestamps_ns >= gt_ns[0]) & (timestamps_ns <= gt_ns[-1])
    clipped = np.clip(timestamps_ns, gt_ns[0], gt_ns[-1])
    source = (gt_ns - gt_ns[0]) * 1e-9
    query = (clipped - gt_ns[0]) * 1e-9
    positions = np.column_stack(
        [np.interp(query, source, raw[:, axis]) for axis in (1, 2, 3)]
    )
    rotations = Slerp(
        source, Rotation.from_quat(raw[:, [5, 6, 7, 4]])
    )(query).as_matrix()
    world_body = np.repeat(np.eye(4)[None], len(query), axis=0)
    world_body[:, :3, :3] = rotations
    world_body[:, :3, 3] = positions
    values = yaml.safe_load(sensor_yaml.read_text())["T_BS"]
    transform = np.asarray(values["data"], dtype=np.float64).reshape(
        int(values["rows"]), int(values["cols"])
    )
    world_camera = world_body @ np.linalg.inv(transform)
    world_camera[~valid] = np.nan
    return world_camera[:, :3, 3]


def evaluate(args: argparse.Namespace) -> None:
    manifest_path = args.output_dir / "manifest_before_gt.json"
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest)
    with np.load(manifest["artifacts"]["candidate"]["path"], allow_pickle=False) as archive:
        candidate = poses_4x4(archive["c2w"])
        orb_mask = np.asarray(archive["orb_mask"], dtype=bool)
        timestamps_ns = np.asarray(archive["timestamps_ns"], dtype=np.int64)
    # GT is opened only after every candidate, raw run, input, policy, and binary hash verifies.
    with np.load(manifest["artifacts"]["cache"]["path"], allow_pickle=False) as archive:
        visual = poses_4x4(archive[manifest["visual_pose_key"]])
        raw_target = (
            np.asarray(archive[args.ground_truth_key], dtype=np.float64)
            if args.ground_truth_key is not None
            else None
        )
    if manifest["visual_convention"] == "w2c":
        visual = np.linalg.inv(visual)
    if raw_target is not None:
        target = raw_target[:, :3, 3] if raw_target.ndim == 3 else raw_target
    elif args.ground_truth_csv is not None and args.camera_sensor_yaml is not None:
        target = external_body_ground_truth(
            args.ground_truth_csv, args.camera_sensor_yaml, timestamps_ns
        )
    else:
        raise ValueError(
            "provide --ground-truth-key or both --ground-truth-csv and "
            "--camera-sensor-yaml"
        )
    result = {
        "protocol": manifest["protocol"],
        "dataset": manifest["dataset"],
        "sequence": manifest["sequence"],
        "manifest_sha256": sha256(manifest_path),
        "visual": metric(visual[:, :3, 3], target),
        "candidate": metric(candidate[:, :3, 3], target),
        "visual_overlap": metric(visual[orb_mask, :3, 3], target[orb_mask]),
        "candidate_overlap": metric(candidate[orb_mask, :3, 3], target[orb_mask]),
    }
    output = args.output_dir / "summary_after_gt.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--dataset", required=True)
    generate_parser.add_argument("--sequence", required=True)
    generate_parser.add_argument("--cache", type=Path, required=True)
    generate_parser.add_argument("--timestamps", type=Path, required=True)
    generate_parser.add_argument("--run-dir", type=Path, action="append", required=True)
    generate_parser.add_argument("--policy", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--visual-pose-key", default="predicted_c2w")
    generate_parser.add_argument("--visual-convention", choices=("c2w", "w2c"), default="c2w")
    generate_parser.add_argument("--timestamp-tolerance-ns", type=int, default=2048)
    generate_parser.add_argument("--sealed-artifact", action="append", default=[])
    generate_parser.add_argument(
        "--protocol",
        default="exploratory candidates sealed before GT; development results observed",
    )
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--ground-truth-key")
    evaluate_parser.add_argument("--ground-truth-csv", type=Path)
    evaluate_parser.add_argument("--camera-sensor-yaml", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
