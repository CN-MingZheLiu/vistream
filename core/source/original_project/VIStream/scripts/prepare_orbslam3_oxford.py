#!/usr/bin/env python3
"""Prepare audited Oxford Spires inputs for ORB-SLAM3 mono-inertial."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp_ns_from_stem(stem: str) -> int:
    return int((Decimal(stem) * Decimal(1_000_000_000)).to_integral_exact())


def full_cache_timestamps(archive: np.lib.npyio.NpzFile) -> tuple[np.ndarray, str]:
    key = "all_timestamps" if "all_timestamps" in archive.files else "timestamps"
    return np.asarray(archive[key], dtype=np.float64), key


def image_set_sha256(paths: list[Path], timestamps_ns: np.ndarray) -> str:
    digest = hashlib.sha256()
    for path, timestamp in zip(paths, timestamps_ns):
        digest.update(f"{int(timestamp)}.png".encode("ascii"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def convert_imu(source: Path, output: Path, first_timestamp_ns: int) -> dict:
    rows = []
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["secs"]) * 1_000_000_000 + int(row["nsecs"])
            rows.append(
                (
                    timestamp,
                    float(row["ang_vel_x"]),
                    float(row["ang_vel_y"]),
                    float(row["ang_vel_z"]),
                    float(row["acc_x"]),
                    float(row["acc_y"]),
                    float(row["acc_z"]),
                )
            )
    if not rows or any(rows[index][0] >= rows[index + 1][0] for index in range(len(rows) - 1)):
        raise ValueError("Oxford IMU timestamps are empty or not strictly increasing")
    prepended = rows[0][0] > first_timestamp_ns
    if prepended:
        rows.insert(0, (first_timestamp_ns - 1, *rows[0][1:]))
    with output.open("w", newline="") as handle:
        handle.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)
    return {
        "source_rows": len(rows) - int(prepended),
        "output_rows": len(rows),
        "prepended_first_measurement": prepended,
        "first_timestamp_ns": rows[0][0],
        "last_timestamp_ns": rows[-1][0],
    }


def camera_to_body(calibration: Path) -> np.ndarray:
    entry = yaml.safe_load(calibration.read_text())["cam0"]
    camera_from_body = np.eye(4)
    camera_from_body[:3, :3] = Rotation.from_quat(entry["C_q_CI"]).as_matrix()
    camera_from_body[:3, 3] = np.asarray(entry["C_r_CI"], dtype=np.float64)
    return np.linalg.inv(camera_from_body)


def settings_text(
    camera: dict,
    imu: dict,
    body_from_camera: np.ndarray,
    resize_width: int,
    resize_height: int,
) -> str:
    matrix = ", ".join(f"{value:.16g}" for value in body_from_camera.reshape(-1))
    projection = camera["projection_matrix"]["data"]
    width = int(camera["image_width"])
    height = int(camera["image_height"])
    return f"""%YAML:1.0

File.version: \"1.0\"
Camera.type: \"PinHole\"
Camera1.fx: {float(projection[0]):.16g}
Camera1.fy: {float(projection[5]):.16g}
Camera1.cx: {width / 2.0:.1f}
Camera1.cy: {height / 2.0:.1f}
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera.width: {width}
Camera.height: {height}
Camera.newWidth: {resize_width}
Camera.newHeight: {resize_height}
Camera.fps: 20
Camera.RGB: 1

IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [{matrix}]
IMU.NoiseGyro: {float(imu['gyroscope_noise_density']):.16g}
IMU.NoiseAcc: {float(imu['accelerometer_noise_density']):.16g}
IMU.GyroWalk: {float(imu['gyroscope_random_walk']):.16g}
IMU.AccWalk: {float(imu['accelerometer_random_walk']):.16g}
IMU.Frequency: {float(imu['update_rate']):.1f}

ORBextractor.nFeatures: 2000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
"""


def prepare(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir()
    with np.load(args.cache, allow_pickle=False) as archive:
        image_paths = [Path(str(value)) for value in archive["image_paths"]]
        cached_timestamps, cached_timestamp_key = full_cache_timestamps(archive)
    if len(image_paths) == 0 or len(image_paths) != len(cached_timestamps):
        raise ValueError("Oxford cache images and timestamps differ")
    if any(not path.is_file() for path in image_paths):
        raise FileNotFoundError(next(path for path in image_paths if not path.is_file()))
    timestamps_ns = np.asarray(
        [timestamp_ns_from_stem(path.stem) for path in image_paths], dtype=np.int64
    )
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError("Oxford image timestamps are not strictly increasing")
    if not np.allclose(
        timestamps_ns.astype(np.float64) * 1e-9,
        cached_timestamps,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("cache timestamps disagree with image filenames")
    for source, timestamp in zip(image_paths, timestamps_ns):
        (image_dir / f"{int(timestamp)}.png").symlink_to(source.resolve())

    times_path = args.output_dir / "timestamps.txt"
    times_path.write_text("\n".join(str(value) for value in timestamps_ns) + "\n")
    imu_path = args.output_dir / "imu_orbslam3.csv"
    imu_diagnostics = convert_imu(args.imu, imu_path, int(timestamps_ns[0]))
    camera = yaml.safe_load(args.camera.read_text())
    imu_config = yaml.safe_load(args.imu_calibration.read_text())
    body_from_camera = camera_to_body(args.extrinsics)
    settings_path = args.output_dir / "Oxford-Spires.yaml"
    settings_path.write_text(
        settings_text(
            camera,
            imu_config,
            body_from_camera,
            args.resize_width,
            args.resize_height,
        )
    )
    diagnostics = {
        "frames": len(image_paths),
        "first_timestamp_ns": int(timestamps_ns[0]),
        "last_timestamp_ns": int(timestamps_ns[-1]),
        "cache_timestamp_key": cached_timestamp_key,
        "maximum_cache_timestamp_error_seconds": float(
            np.max(np.abs(timestamps_ns.astype(np.float64) * 1e-9 - cached_timestamps))
        ),
        "imu": imu_diagnostics,
        "resize": [args.resize_width, args.resize_height],
        "body_from_camera": body_from_camera.tolist(),
    }
    diagnostics_path = args.output_dir / "prepare_before_gt.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    manifest = {
        "protocol": "Oxford ORB-SLAM3 inputs prepared and hashed without GT access",
        "cache": str(args.cache.resolve()),
        "cache_sha256": sha256(args.cache),
        "source_images_sha256": image_set_sha256(image_paths, timestamps_ns),
        "image_dir": str(image_dir.resolve()),
        "timestamps": str(times_path.resolve()),
        "timestamps_sha256": sha256(times_path),
        "source_imu": str(args.imu.resolve()),
        "source_imu_sha256": sha256(args.imu),
        "converted_imu": str(imu_path.resolve()),
        "converted_imu_sha256": sha256(imu_path),
        "camera": str(args.camera.resolve()),
        "camera_sha256": sha256(args.camera),
        "imu_calibration": str(args.imu_calibration.resolve()),
        "imu_calibration_sha256": sha256(args.imu_calibration),
        "extrinsics": str(args.extrinsics.resolve()),
        "extrinsics_sha256": sha256(args.extrinsics),
        "settings": str(settings_path.resolve()),
        "settings_sha256": sha256(settings_path),
        "diagnostics": str(diagnostics_path.resolve()),
        "diagnostics_sha256": sha256(diagnostics_path),
    }
    (args.output_dir / "manifest_before_gt.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(diagnostics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--imu", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--imu-calibration", type=Path, required=True)
    parser.add_argument("--extrinsics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resize-width", type=int, default=720)
    parser.add_argument("--resize-height", type=int, default=540)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
