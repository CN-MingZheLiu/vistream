#!/usr/bin/env python3
"""Build a leakage-free TartanAir window split matched to KITTI driving."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from vistream.data import discover_tartanair_sequences


# These environments were selected by inspecting the installed TartanAir
# images.  Training covers daylight urban, industrial, suburban, and rural
# appearance.  Validation holds out one urban, one suburban, and one rural
# environment; no KITTI image, pose, or metric is used to build this split.
TRAIN_ENVIRONMENTS = (
    "ConstructionSite",
    "GreatMarsh",
    "ModUrbanCity",
    "ModernCityDowntown",
    "NordicHarbor",
    "OldIndustrialCity",
    "OldTownSummer",
    "SeasideTown",
    "SeasonalForestAutumn",
    "SeasonalForestSpring",
    "VictorianStreet",
    "WesternDesertTown",
)
VALIDATION_ENVIRONMENTS = (
    "DesertGasStation",
    "Downtown",
    "ModularNeighborhood",
)

CRITERIA = {
    "difficulty": "easy",
    "median_tilt_deg_max": 25.0,
    "p90_tilt_deg_max": 38.0,
    "forward_path_fraction_min": 0.45,
    "vertical_path_fraction_max": 0.25,
    "net_to_path_ratio_min": 0.50,
    "p90_angular_speed_rad_s_max": 1.15,
    "path_length_m_min": 1.0,
    # KITTI 05 averages about 7.7 m/s. The v1 split averaged only 2.3 m/s,
    # which taught the pose head the wrong per-frame metric scale.
    "mean_speed_m_s_min": 3.0,
    "mean_speed_m_s_max": 15.0,
    "median_speed_m_s_min": 2.0,
    "target_speed_m_s": 7.5,
}


def _rotation_matrices(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.clip(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12, None
    )
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(-1, 3, 3)


def _starts(frames: int, window_frames: int, stride_frames: int) -> list[int]:
    starts = list(range(0, max(0, frames - window_frames + 1), stride_frames))
    if frames >= window_frames:
        last = frames - window_frames
        if not starts or starts[-1] != last:
            starts.append(last)
    return starts


def _metrics(
    pose: np.ndarray, camera_time: np.ndarray, rotations: np.ndarray
) -> dict[str, float] | None:
    displacement = np.diff(pose[:, :3], axis=0)
    distance = np.linalg.norm(displacement, axis=-1)
    path_length = float(distance.sum())
    dt = np.diff(camera_time)
    if path_length <= 1e-6 or np.any(dt <= 1e-5) or np.any(dt >= 1.0):
        return None

    local_displacement = np.einsum(
        "nij,nj->ni", rotations[:-1].transpose(0, 2, 1), displacement
    )
    tilt = np.degrees(
        np.arccos(np.abs(rotations[:, 2, 2]).clip(0.0, 1.0))
    )
    quaternion = pose[:, 3:7]
    quaternion = quaternion / np.clip(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12, None
    )
    quaternion_dot = np.abs(
        np.sum(quaternion[:-1] * quaternion[1:], axis=-1)
    ).clip(0.0, 1.0)
    angular_speed = 2.0 * np.arccos(quaternion_dot) / dt
    frame_speed = distance / dt
    values = {
        "median_tilt_deg": float(np.median(tilt)),
        "p90_tilt_deg": float(np.quantile(tilt, 0.9)),
        "forward_path_fraction": float(
            np.abs(local_displacement[:, 0]).sum() / path_length
        ),
        "vertical_path_fraction": float(
            np.abs(displacement[:, 2]).sum() / path_length
        ),
        "net_to_path_ratio": float(
            np.linalg.norm(pose[-1, :3] - pose[0, :3]) / path_length
        ),
        "p90_angular_speed_rad_s": float(np.quantile(angular_speed, 0.9)),
        "path_length_m": path_length,
        "mean_speed_m_s": float(distance.sum() / dt.sum()),
        "median_speed_m_s": float(np.median(frame_speed)),
        "p90_speed_m_s": float(np.quantile(frame_speed, 0.9)),
    }
    return values


def _accepted(values: dict[str, float]) -> bool:
    return (
        values["median_tilt_deg"] <= CRITERIA["median_tilt_deg_max"]
        and values["p90_tilt_deg"] <= CRITERIA["p90_tilt_deg_max"]
        and values["forward_path_fraction"] >= CRITERIA["forward_path_fraction_min"]
        and values["vertical_path_fraction"] <= CRITERIA["vertical_path_fraction_max"]
        and values["net_to_path_ratio"] >= CRITERIA["net_to_path_ratio_min"]
        and values["p90_angular_speed_rad_s"]
        <= CRITERIA["p90_angular_speed_rad_s_max"]
        and values["path_length_m"] >= CRITERIA["path_length_m_min"]
        and values["mean_speed_m_s"] >= CRITERIA["mean_speed_m_s_min"]
        and values["mean_speed_m_s"] <= CRITERIA["mean_speed_m_s_max"]
        and values["median_speed_m_s"] >= CRITERIA["median_speed_m_s_min"]
    )


def _score(values: dict[str, float]) -> float:
    """Lower is closer to upright, forward, planar vehicle motion."""
    return (
        values["median_tilt_deg"] / CRITERIA["median_tilt_deg_max"]
        + values["p90_tilt_deg"] / CRITERIA["p90_tilt_deg_max"]
        + (1.0 - values["forward_path_fraction"])
        + values["vertical_path_fraction"] / CRITERIA["vertical_path_fraction_max"]
        + (1.0 - values["net_to_path_ratio"])
        + values["p90_angular_speed_rad_s"]
        / CRITERIA["p90_angular_speed_rad_s_max"]
        + abs(values["mean_speed_m_s"] - CRITERIA["target_speed_m_s"])
        / CRITERIA["target_speed_m_s"]
    )


def _sequence_path(root: Path, sequence: str) -> Path:
    environment, difficulty, trajectory = sequence.split("/")
    return root / environment / f"Data_{difficulty}" / trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", default="/data2/huangshuaixin/tartanair-v2"
    )
    parser.add_argument(
        "--output", default="splits/tartanair_kitti_matched_v2.json"
    )
    parser.add_argument("--window-frames", type=int, default=16)
    parser.add_argument("--stride-frames", type=int, default=8)
    parser.add_argument("--max-train-per-environment", type=int, default=1000)
    parser.add_argument(
        "--train-window-multiple",
        type=int,
        default=64,
        help="Trim the worst-matched tail for exact 4-GPU, BS4, accum4 epochs.",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    target_environments = set(TRAIN_ENVIRONMENTS) | set(VALIDATION_ENVIRONMENTS)
    candidates: dict[str, list[tuple[float, str, int, float]]] = defaultdict(list)
    for sequence in discover_tartanair_sequences(root):
        environment, difficulty, _ = sequence.split("/")
        if difficulty != CRITERIA["difficulty"] or environment not in target_environments:
            continue
        path = _sequence_path(root, sequence)
        pose = np.loadtxt(path / "pose_lcam_front.txt", dtype=np.float64)
        camera_time = np.loadtxt(
            path / "imu" / "cam_time.txt", dtype=np.float64
        ).reshape(-1)
        image_count = sum(1 for _ in (path / "image_lcam_front").glob("*.png"))
        depth_count = sum(1 for _ in (path / "depth_lcam_front").glob("*.png"))
        frames = min(len(pose), len(camera_time), image_count, depth_count)
        pose = np.atleast_2d(pose)[:frames]
        camera_time = camera_time[:frames]
        rotations = _rotation_matrices(pose[:, 3:7])
        for start in _starts(frames, args.window_frames, args.stride_frames):
            stop = start + args.window_frames
            values = _metrics(
                pose[start:stop], camera_time[start:stop], rotations[start:stop]
            )
            if values is not None and _accepted(values):
                candidates[environment].append(
                    (_score(values), sequence, start, values["mean_speed_m_s"])
                )

    train_records: list[tuple[float, str, int, float]] = []
    validation_records: list[tuple[float, str, int, float]] = []
    candidate_counts: dict[str, int] = {}
    for environment in sorted(target_environments):
        records = sorted(candidates[environment])
        candidate_counts[environment] = len(records)
        if environment in TRAIN_ENVIRONMENTS:
            records = records[: args.max_train_per_environment]
            train_records.extend(records)
        else:
            validation_records.extend(records)

    if not train_records or not validation_records:
        raise RuntimeError("KITTI-matched selection produced an empty split")
    if args.train_window_multiple < 1:
        raise ValueError("--train-window-multiple must be positive")
    retained = len(train_records) - len(train_records) % args.train_window_multiple
    if retained == 0:
        raise RuntimeError("train-window multiple is larger than the selected split")
    train_records = sorted(train_records)[:retained]
    train_records.sort(key=lambda item: (item[1], item[2]))
    validation_records.sort(key=lambda item: (item[1], item[2]))
    selected_counts = Counter(
        sequence.split("/", 1)[0]
        for _, sequence, _, _ in train_records + validation_records
    )
    payload = {
        "format": "vistream-tartanair-window-split-v1",
        "profile": "kitti-matched-v2",
        "data_root": str(root.resolve()),
        "window_frames": args.window_frames,
        "stride_frames": args.stride_frames,
        "policy": {
            "description": (
                "Daylight outdoor environments plus pose-only filtering for "
                "upright, planar, forward, smooth vehicle-like motion plus a "
                "KITTI-like speed range. No KITTI "
                "sample, label, prediction, or metric is used."
            ),
            "criteria": CRITERIA,
            "max_train_per_environment": args.max_train_per_environment,
            "train_window_multiple": args.train_window_multiple,
            "train_environments": list(TRAIN_ENVIRONMENTS),
            "validation_environments": list(VALIDATION_ENVIRONMENTS),
        },
        "summary": {
            "train_windows": len(train_records),
            "validation_windows": len(validation_records),
            "candidate_counts_by_environment": candidate_counts,
            "selected_counts_by_environment": dict(sorted(selected_counts.items())),
            "mean_train_speed_m_s": float(
                np.mean([speed for _, _, _, speed in train_records])
            ),
            "mean_validation_speed_m_s": float(
                np.mean([speed for _, _, _, speed in validation_records])
            ),
        },
        "train": [
            {"sequence": sequence, "start": start}
            for _, sequence, start, _ in train_records
        ],
        "validation": [
            {"sequence": sequence, "start": start}
            for _, sequence, start, _ in validation_records
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
