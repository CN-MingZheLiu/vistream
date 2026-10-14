"""KITTI odometry reader with explicit raw-OXTS synchronization requirement."""

from __future__ import annotations

import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


KITTI_ODOMETRY_TO_RAW = {
    "00": "2011_10_03_drive_0027_sync",
    "01": "2011_10_03_drive_0042_sync",
    "02": "2011_10_03_drive_0034_sync",
    "03": "2011_09_26_drive_0067_sync",
    "04": "2011_09_30_drive_0016_sync",
    "05": "2011_09_30_drive_0018_sync",
    "06": "2011_09_30_drive_0020_sync",
    "07": "2011_09_30_drive_0027_sync",
    "08": "2011_09_30_drive_0028_sync",
    "09": "2011_09_30_drive_0033_sync",
    "10": "2011_09_30_drive_0034_sync",
}

# The odometry split uses the rectified camera-0 frame, while raw OXTS is in
# the IMU frame.  These dates are also the names used by KITTI's calibration
# archives (``<date>_calib.zip``).
KITTI_ODOMETRY_TO_DATE = {
    "00": "2011_10_03",
    "01": "2011_10_03",
    "02": "2011_10_03",
    "03": "2011_09_26",
    "04": "2011_09_30",
    "05": "2011_09_30",
    "06": "2011_09_30",
    "07": "2011_09_30",
    "08": "2011_09_30",
    "09": "2011_09_30",
    "10": "2011_09_30",
}

# KITTI's released IMU->Velodyne calibration is shared by the odometry raw
# drives.  Keep it as a verified fallback because many lightweight raw mirrors
# contain only OXTS packets and omit the 4 KB calibration archive.
KITTI_IMU_TO_VELO = np.array(
    (
        (9.999976e-01, 7.553071e-04, -2.035826e-03, -8.086759e-01),
        (-7.854027e-04, 9.998898e-01, -1.482298e-02, 3.195559e-01),
        (2.024406e-03, 1.482454e-02, 9.998881e-01, -7.997231e-01),
    ),
    dtype=np.float64,
)


def kitti_raw_sequence_for_odometry(
    raw_root: str | Path, sequence: str
) -> Path:
    sequence = f"{int(sequence):02d}"
    name = KITTI_ODOMETRY_TO_RAW.get(sequence)
    if name is None:
        raise ValueError(f"KITTI odometry sequence {sequence} has no public raw mapping")
    direct = Path(raw_root) / name
    if direct.is_dir():
        return direct
    nested = Path(raw_root) / name[:10] / name
    if nested.is_dir():
        return nested
    raise FileNotFoundError(
        f"raw KITTI OXTS is required for odometry {sequence}; expected {direct}"
    )


def verify_kitti_imu_coverage(raw_root: str | Path, sequences: list[str]) -> dict[str, str]:
    available, missing = {}, []
    for sequence in sequences:
        try:
            available[sequence] = str(kitti_raw_sequence_for_odometry(raw_root, sequence))
        except (ValueError, FileNotFoundError) as error:
            missing.append(f"{sequence}: {error}")
    if missing:
        raise FileNotFoundError("KITTI fusion evaluation cannot proceed without raw IMU: " + " | ".join(missing))
    return available


def _parse_named_calibration(payload: str) -> dict[str, np.ndarray]:
    """Parse KITTI raw calibration files containing ``R:`` and ``T:`` lines."""
    result: dict[str, np.ndarray] = {}
    for line in payload.splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        values = np.fromstring(values, sep=" ")
        if key in {"R", "T"} and values.size:
            result[key] = values
    if "R" not in result or "T" not in result:
        raise ValueError("KITTI calibration must contain R and T")
    if result["R"].size != 9 or result["T"].size != 3:
        raise ValueError("KITTI calibration has invalid R/T dimensions")
    return result


def _calibration_payloads(raw_root: Path, raw_path: Path, date: str) -> list[tuple[str, str]]:
    """Return available raw calibration payloads without assuming extraction."""
    candidates: list[tuple[str, str]] = []
    roots = (raw_path, raw_path.parent, raw_root / date)
    for root in roots:
        path = root / "calib_imu_to_velo.txt"
        if path.is_file():
            candidates.append(("imu_to_velo", path.read_text()))
        path = root / "calib_velo_to_cam.txt"
        if path.is_file():
            candidates.append(("velo_to_cam", path.read_text()))
    archive_candidates = (
        raw_root / f"{date}_calib.zip",
        raw_root / f"{date}_calib" / f"{date}_calib.zip",
    )
    for archive in archive_candidates:
        if not archive.is_file():
            continue
        with zipfile.ZipFile(archive) as stream:
            for stem, name in (
                ("imu_to_velo", f"{date}/calib_imu_to_velo.txt"),
                ("velo_to_cam", f"{date}/calib_velo_to_cam.txt"),
            ):
                try:
                    candidates.append((stem, stream.read(name).decode("ascii")))
                except KeyError:
                    continue
    return candidates


def _load_camera_from_imu(
    raw_root: str | Path,
    raw_path: Path,
    sequence: str,
    velo_to_camera: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Compose the rectified camera-from-IMU transform from raw calibration.

    ``sequences/<seq>/calib.txt`` only stores Velodyne->rectified-camera.  A
    previous implementation used that rotation directly for OXTS, silently
    introducing an axis/mounting error.  Exact raw calibration is preferred;
    the fallback retains compatibility with partial KITTI mirrors and labels
    the result so callers can reject it for final metrics.
    """
    root = Path(raw_root)
    date = KITTI_ODOMETRY_TO_DATE.get(sequence)
    if date is None:
        return velo_to_camera, "odometry_velo_to_camera_approximate"
    parsed: dict[str, np.ndarray] = {}
    for stem, payload in _calibration_payloads(root, raw_path, date):
        try:
            parsed[stem] = _parse_named_calibration(payload)
        except ValueError:
            continue
    imu_to_velo = np.eye(4, dtype=np.float64)
    if "imu_to_velo" in parsed:
        imu_to_velo[:3, :3] = parsed["imu_to_velo"]["R"].reshape(3, 3)
        imu_to_velo[:3, 3] = parsed["imu_to_velo"]["T"]
        source = "raw_imu_to_velo_composed"
    else:
        imu_to_velo[:3, :] = KITTI_IMU_TO_VELO
        source = "official_imu_to_velo_builtin"
    camera_from_imu = np.eye(4, dtype=np.float64)
    camera_from_imu[:3, :] = velo_to_camera @ imu_to_velo
    return camera_from_imu[:3], source


def _parse_timestamps(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text().splitlines():
        stamp = datetime.strptime(line[:26], "%Y-%m-%d %H:%M:%S.%f")
        values.append(stamp.timestamp())
    return np.asarray(values, dtype=np.float64)


def _load_oxts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted((path / "oxts" / "data").glob("*.txt"))
    timestamps_path = path / "oxts" / "timestamps.txt"
    if not files or not timestamps_path.is_file():
        raise FileNotFoundError(f"KITTI OXTS files are incomplete under {path}")
    times = _parse_timestamps(timestamps_path)
    count = min(len(files), len(times))
    values = np.stack([np.loadtxt(item, dtype=np.float64) for item in files[:count]])
    imu = np.concatenate((values[:, 11:14], values[:, 17:20]), axis=-1)
    return imu.astype(np.float32), times[:count]


def _find_extract_sequence(raw_root: str | Path, sync_path: Path) -> Path | None:
    extract_name = sync_path.name.replace("_sync", "_extract")
    root = Path(raw_root)
    date = sync_path.name[:10]
    candidates = (
        sync_path.with_name(extract_name),
        root / extract_name,
        root / date / extract_name,
    )
    for candidate in candidates:
        if (candidate / "oxts" / "timestamps.txt").is_file():
            return candidate
    return None


def _load_square_image(path: Path, size: int) -> tuple[torch.Tensor, float, float, float]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = max(width, height)
    scale = size / side
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0
    pad_h, pad_w = size - tensor.shape[-2], size - tensor.shape[-1]
    top, left = pad_h // 2, pad_w // 2
    # LingBot's released preprocessing pads with white (1.0). Black bars are
    # a large out-of-distribution region for KITTI's wide images.
    tensor = F.pad(
        tensor,
        (left, pad_w - left, top, pad_h - top),
        value=1.0,
    )
    return tensor, scale, float(left), float(top)


class KITTIOdometryWindowDataset(Dataset):
    """KITTI RGB/pose windows paired one-to-one with raw OXTS IMU."""

    def __init__(
        self,
        odometry_root: str | Path,
        raw_root: str | Path,
        sequence: str,
        image_size: int = 518,
        window_frames: int = 16,
        stride_frames: int = 8,
        max_imu_samples: int = 32,
        prefer_high_rate_imu: bool = True,
    ) -> None:
        self.odometry_root = Path(odometry_root)
        self.raw_path = kitti_raw_sequence_for_odometry(raw_root, sequence)
        self.sequence = f"{int(sequence):02d}"
        self.image_size = image_size
        self.window_frames = window_frames
        self.max_imu_samples = max(1, int(max_imu_samples))
        sequence_path = self.odometry_root / "sequences" / self.sequence
        self.image_files = sorted((sequence_path / "image_2").glob("*.png"))
        pose_path = self.odometry_root / "poses" / f"{self.sequence}.txt"
        if not pose_path.is_file():
            raise FileNotFoundError(f"KITTI sequence {self.sequence} has no ground-truth poses")
        pose_values = np.loadtxt(pose_path, dtype=np.float32).reshape(-1, 3, 4)
        self.poses = torch.eye(4).repeat(len(pose_values), 1, 1)
        self.poses[:, :3, :] = torch.from_numpy(pose_values)
        sync_imu, sync_times = _load_oxts(self.raw_path)
        count = min(len(self.image_files), len(self.poses), len(sync_imu))
        if count < window_frames:
            raise RuntimeError(f"KITTI {self.sequence} has only {count} aligned raw frames")
        self.image_files = self.image_files[:count]
        self.poses = self.poses[:count]
        self.times = sync_times[:count]
        self.sync_imu = sync_imu[:count]
        extract_path = (
            _find_extract_sequence(raw_root, self.raw_path)
            if prefer_high_rate_imu
            else None
        )
        if extract_path is not None:
            self.imu, self.imu_times = _load_oxts(extract_path)
            self.imu_source = "extract"
        else:
            self.imu, self.imu_times = self.sync_imu, self.times
            self.imu_source = "sync"
        imu_intervals = np.diff(self.imu_times)
        positive_intervals = imu_intervals[imu_intervals > 0]
        self.imu_rate_hz = (
            float(1.0 / np.median(positive_intervals))
            if positive_intervals.size
            else 0.0
        )
        initial_dt = float(np.median(np.diff(self.times[: min(32, len(self.times))])))
        interval_begins = np.concatenate(([self.times[0] - initial_dt], self.times[:-1]))
        interval_left = np.searchsorted(self.imu_times, interval_begins, side="right")
        interval_right = np.searchsorted(self.imu_times, self.times, side="right")
        interval_counts = np.maximum(interval_right - interval_left, 1)
        self.max_interval_imu_samples = int(interval_counts.max())
        if self.max_interval_imu_samples > self.max_imu_samples:
            raise RuntimeError(
                f"KITTI {self.sequence} requires {self.max_interval_imu_samples} IMU "
                f"samples per RGB interval but max_imu_samples={self.max_imu_samples}; "
                "increase the capacity to keep the sequence lossless"
            )
        calibration = {}
        for line in (sequence_path / "calib.txt").read_text().splitlines():
            key, values = line.split(":", 1)
            calibration[key] = np.fromstring(values, sep=" ")
        projection = calibration["P2"].reshape(3, 4)
        with Image.open(self.image_files[0]) as image:
            width, height = image.size
        side = max(width, height)
        scale = image_size / side
        left, top = (image_size - round(width * scale)) / 2, (image_size - round(height * scale)) / 2
        self.intrinsics = torch.tensor(
            ((projection[0, 0] * scale, 0, projection[0, 2] * scale + left),
             (0, projection[1, 1] * scale, projection[1, 2] * scale + top),
             (0, 0, 1)), dtype=torch.float32
        ).expand(count, -1, -1).clone()
        velo_to_camera = calibration["Tr"].reshape(3, 4)
        camera_from_imu, self.imu_calibration_source = _load_camera_from_imu(
            raw_root, self.raw_path, self.sequence, velo_to_camera
        )
        self.camera_from_imu = torch.from_numpy(camera_from_imu.astype(np.float32))
        self.rotation_camera_from_imu = self.camera_from_imu[:3, :3]
        self.gravity_camera = self.rotation_camera_from_imu @ torch.from_numpy(
            self.sync_imu[: min(20, len(self.sync_imu)), :3].mean(axis=0)
        )
        self.velodyne_files = sorted((self.raw_path / "velodyne_points" / "data").glob("*.bin"))[:count]
        self.samples = list(range(0, count - window_frames + 1, stride_frames))
        last = count - window_frames
        if not self.samples or self.samples[-1] != last:
            self.samples.append(last)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        start = self.samples[index]
        stop = start + self.window_frames
        images = torch.stack([
            _load_square_image(path, self.image_size)[0] for path in self.image_files[start:stop]
        ])
        absolute_times = self.times[start:stop]
        camera_times = torch.from_numpy((absolute_times - absolute_times[0]).astype(np.float32))
        initial_dt = float(np.median(np.diff(self.times[: min(32, len(self.times))])))
        previous_time = self.times[start - 1] if start else absolute_times[0] - initial_dt
        dt = np.diff(absolute_times, prepend=previous_time).astype(np.float32)
        imu = torch.zeros(self.window_frames, self.max_imu_samples, 6)
        imu_dt = torch.zeros(self.window_frames, self.max_imu_samples)
        imu_mask = torch.zeros(self.window_frames, self.max_imu_samples, dtype=torch.bool)
        for local, end in enumerate(absolute_times):
            begin = previous_time if local == 0 else absolute_times[local - 1]
            left = int(np.searchsorted(self.imu_times, begin, side="right"))
            right = int(np.searchsorted(self.imu_times, end, side="right"))
            ids = np.arange(left, right, dtype=np.int64)
            if ids.size == 0:
                ids = np.array(
                    [min(max(right - 1, 0), len(self.imu_times) - 1)],
                    dtype=np.int64,
                )
            if ids.size > self.max_imu_samples:
                raise RuntimeError(
                    f"KITTI {self.sequence} interval {begin:.6f}->{end:.6f} has "
                    f"{ids.size} IMU samples, exceeding max_imu_samples="
                    f"{self.max_imu_samples}; increase the capacity to keep every sample"
                )
            sample_times = np.clip(self.imu_times[ids], begin, end)
            sample_dt = np.diff(sample_times, prepend=begin)
            sample_dt[-1] += max(float(end - sample_times[-1]), 0.0)
            duration = max(float(end - begin), 1e-6)
            if sample_dt.sum() > 0:
                sample_dt *= duration / sample_dt.sum()
            else:
                sample_dt[-1] = duration
            count = len(ids)
            imu[local, :count] = torch.from_numpy(self.imu[ids])
            imu_dt[local, :count] = torch.from_numpy(sample_dt.astype(np.float32))
            imu_mask[local, :count] = True
        return {
            "images": images,
            "depth": torch.empty(0),
            "depth_mask": torch.empty(0, dtype=torch.bool),
            "poses": self.poses[start:stop].clone(),
            "gt_intrinsics": self.intrinsics[start:stop].clone(),
            "camera_times": camera_times,
            "camera_dt": torch.from_numpy(dt),
            "imu": imu,
            "imu_dt": imu_dt,
            "imu_mask": imu_mask,
            "rotation_camera_from_imu": self.rotation_camera_from_imu.clone(),
            "gravity_camera": self.gravity_camera.clone(),
            "imu_rate_hz": torch.tensor(self.imu_rate_hz, dtype=torch.float32),
            "imu_source": self.imu_source,
            "imu_calibration_source": self.imu_calibration_source,
            "sequence": self.sequence,
            "start_frame": torch.tensor(start),
        }

    def load_lidar_camera(self, frame: int) -> np.ndarray:
        """Load a sparse metric ground-truth cloud in camera coordinates."""
        if frame >= len(self.velodyne_files):
            return np.empty((0, 3), dtype=np.float32)
        points = np.fromfile(self.velodyne_files[frame], dtype=np.float32).reshape(-1, 4)[:, :3]
        sequence_path = self.odometry_root / "sequences" / self.sequence
        tr_line = next(line for line in (sequence_path / "calib.txt").read_text().splitlines() if line.startswith("Tr:"))
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :] = np.fromstring(tr_line.split(":", 1)[1], sep=" ").reshape(3, 4)
        homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float32)), axis=-1)
        camera = (transform @ homogeneous.T).T[:, :3]
        return camera[camera[:, 2] > 0]
