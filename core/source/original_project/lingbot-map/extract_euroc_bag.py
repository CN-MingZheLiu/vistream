#!/usr/bin/env python3
"""Extract the EuRoC cam0/IMU streams needed by the fixed-lag evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml
from rosbags.highlevel import AnyReader


T_CAM0_IMU = [
    [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
    [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
    [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
    [0.0, 0.0, 0.0, 1.0],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--converted-gt", type=Path,
        default=Path("/data2/huangshuaixin/kiss_imu_data/EUROC/MH_01_easy/gt_pose.csv"),
        help="Existing EuRoC GT export with timestamps relative to the first IMU sample.",
    )
    return parser.parse_args()


def image_array(message) -> np.ndarray:
    height, width, step = int(message.height), int(message.width), int(message.step)
    raw = np.asarray(message.data, dtype=np.uint8).reshape(height, step)
    encoding = str(message.encoding).lower()
    if encoding in {"mono8", "8uc1"}:
        return raw[:, :width]
    if encoding in {"rgb8", "bgr8"}:
        image = raw[:, : 3 * width].reshape(height, width, 3)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else image
    raise ValueError(f"Unsupported EuRoC image encoding {message.encoding!r}")


def sensor_timestamp_ns(message, fallback: int) -> int:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return int(fallback)
    seconds = getattr(stamp, "sec", None)
    nanoseconds = getattr(stamp, "nanosec", getattr(stamp, "nsec", None))
    if seconds is None or nanoseconds is None:
        return int(fallback)
    return int(seconds) * 1_000_000_000 + int(nanoseconds)


def write_calibration(root: Path) -> None:
    cam = {
        "sensor_type": "camera", "comment": "EuRoC MAV cam0",
        "T_BS": {"rows": 4, "cols": 4, "data": T_CAM0_IMU},
        "rate_hz": 20, "resolution": [752, 480], "camera_model": "pinhole",
        "intrinsics": [458.654, 457.296, 367.215, 248.375],
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05],
    }
    imu = {
        "sensor_type": "imu", "comment": "EuRoC MAV ADIS16448",
        "rate_hz": 200,
        "gyroscope_noise_density": 1.6968e-4,
        "gyroscope_random_walk": 1.9393e-5,
        "accelerometer_noise_density": 2.0e-3,
        "accelerometer_random_walk": 3.0e-3,
    }
    for path, values in (
        (root / "mav0/cam0/sensor.yaml", cam),
        (root / "mav0/imu0/sensor.yaml", imu),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(values, handle, sort_keys=False)


def write_ground_truth(root: Path, converted_path: Path, imu_origin_ns: int) -> None:
    converted = np.loadtxt(converted_path, delimiter=",", dtype=np.float64)
    if converted.ndim != 2 or converted.shape[1] < 8:
        raise ValueError(f"Unexpected converted GT shape {converted.shape}")
    output = root / "mav0/state_groundtruth_estimate0/data.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamps_ns = imu_origin_ns + np.rint(converted[:, 0] * 1e9).astype(np.int64)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "#timestamp", "p_RS_R_x", "p_RS_R_y", "p_RS_R_z",
            "q_RS_w", "q_RS_x", "q_RS_y", "q_RS_z",
        ])
        for timestamp, row in zip(timestamps_ns, converted):
            writer.writerow([int(timestamp), *row[1:8].tolist()])


def main() -> None:
    args = parse_args()
    if not args.bag.is_file():
        raise FileNotFoundError(args.bag)
    image_root = args.output_root / "mav0/cam0/data"
    image_root.mkdir(parents=True, exist_ok=True)
    camera_rows: list[tuple[int, str]] = []
    imu_rows: list[list[float | int]] = []
    with AnyReader([args.bag]) as reader:
        available = {connection.topic: connection.msgtype for connection in reader.connections}
        image_connections = [
            connection for connection in reader.connections
            if connection.topic.rstrip("/").endswith("cam0/image_raw")
        ]
        imu_connections = [
            connection for connection in reader.connections
            if connection.topic.rstrip("/").endswith("imu0")
        ]
        if not image_connections or not imu_connections:
            raise ValueError(
                f"Required cam0/imu0 topics not found. Available topics: {available}"
            )
        selected = image_connections + imu_connections
        for connection, timestamp, raw in reader.messages(connections=selected):
            message = reader.deserialize(raw, connection.msgtype)
            timestamp = sensor_timestamp_ns(message, timestamp)
            if connection in image_connections:
                filename = f"{timestamp}.png"
                if not cv2.imwrite(str(image_root / filename), image_array(message)):
                    raise OSError(f"Failed to write {image_root / filename}")
                camera_rows.append((timestamp, filename))
                if len(camera_rows) % 250 == 0:
                    print(f"Extracted {len(camera_rows)} cam0 frames", flush=True)
            else:
                imu_rows.append([
                    timestamp,
                    message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z,
                    message.linear_acceleration.x, message.linear_acceleration.y,
                    message.linear_acceleration.z,
                ])
    camera_rows.sort()
    imu_rows.sort(key=lambda row: row[0])
    camera_csv = args.output_root / "mav0/cam0/data.csv"
    with camera_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["#timestamp [ns]", "filename"])
        writer.writerows(camera_rows)
    imu_csv = args.output_root / "mav0/imu0/data.csv"
    imu_csv.parent.mkdir(parents=True, exist_ok=True)
    with imu_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]",
            "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]",
            "a_RS_S_z [m s^-2]",
        ])
        writer.writerows(imu_rows)
    write_calibration(args.output_root)
    write_ground_truth(args.output_root, args.converted_gt, int(imu_rows[0][0]))
    print(
        f"Wrote {len(camera_rows)} cam0 frames and {len(imu_rows)} IMU samples "
        f"to {args.output_root}", flush=True,
    )


if __name__ == "__main__":
    main()
