"""Timestamp-aligned TartanAir v2 RGB/depth/pose/IMU windows."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..geometry.lie import quaternion_xyzw_to_matrix


def discover_tartanair_sequences(
    data_root: str | Path,
    environments: Optional[Iterable[str]] = None,
) -> list[str]:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"TartanAir root not found: {root}")
    allowed = set(environments) if environments is not None else None
    sequences: list[str] = []
    for environment in sorted(root.iterdir()):
        if not environment.is_dir() or environment.name in {"raw", "tartan-air-tool"}:
            continue
        if allowed is not None and environment.name not in allowed:
            continue
        for difficulty in ("easy", "hard"):
            parent = environment / f"Data_{difficulty}"
            if not parent.is_dir():
                continue
            for trajectory in sorted(parent.iterdir()):
                required = (
                    trajectory / "image_lcam_front",
                    trajectory / "pose_lcam_front.txt",
                    trajectory / "imu" / "imu_time.txt",
                    trajectory / "imu" / "cam_time.txt",
                )
                if trajectory.is_dir() and required[0].is_dir() and all(path.is_file() for path in required[1:]):
                    sequences.append(f"{environment.name}/{difficulty}/{trajectory.name}")
    if not sequences:
        raise RuntimeError(f"no complete TartanAir sequences found under {root}")
    return sequences


def split_tartanair_by_environment(
    sequences: Sequence[str],
    validation_environments: int = 15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Deterministically split all trajectories from an environment together."""
    by_environment: dict[str, list[str]] = {}
    for sequence in sequences:
        by_environment.setdefault(sequence.split("/", 1)[0], []).append(sequence)
    environments = sorted(by_environment)
    if len(environments) < 2:
        raise ValueError("environment-level split needs at least two environments")
    validation_environments = min(
        max(1, int(validation_environments)), len(environments) - 1
    )
    shuffled = list(environments)
    random.Random(seed).shuffle(shuffled)
    val_names = set(shuffled[:validation_environments])
    train = [sequence for sequence in sequences if sequence.split("/", 1)[0] not in val_names]
    validation = [sequence for sequence in sequences if sequence.split("/", 1)[0] in val_names]
    return train, validation


def _decode_tartanair_depth(path: Path) -> np.ndarray:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    if rgba.shape[-1] != 4:
        raise ValueError(f"depth file is not RGBA: {path}")
    return np.ascontiguousarray(rgba).view("<f4").reshape(rgba.shape[:2]).astype(np.float32)


def _load_rgb(path: Path, image_size: int) -> torch.Tensor:
    array = np.array(Image.open(path).convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return F.interpolate(
        tensor.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False
    ).squeeze(0)


def _load_depth(path: Path, image_size: int) -> torch.Tensor:
    depth = torch.from_numpy(_decode_tartanair_depth(path)).unsqueeze(0).unsqueeze(0)
    depth = F.interpolate(depth, size=(image_size, image_size), mode="nearest").squeeze(0)
    return depth


def _pose_to_matrix(pose: np.ndarray) -> torch.Tensor:
    if pose.shape[-1] != 7:
        raise ValueError("TartanAir pose must be tx ty tz qx qy qz qw")
    translation = torch.from_numpy(pose[..., :3].astype(np.float32))
    quaternion = torch.from_numpy(pose[..., 3:7].astype(np.float32))
    rotation = quaternion_xyzw_to_matrix(quaternion)
    transform = torch.zeros(pose.shape[:-1] + (4, 4), dtype=torch.float32)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


class TartanAirWindowDataset(Dataset):
    """Return fixed RGB windows and variable-rate IMU samples aligned by time."""

    def __init__(
        self,
        data_root: str | Path,
        sequences: Sequence[str],
        window_frames: int = 16,
        stride_frames: int = 8,
        image_size: int = 518,
        max_imu_samples: int = 32,
        include_depth: bool = True,
        acc_source: str = "acc",
        cache_size: int = 4,
        samples: Optional[Sequence[Tuple[str, int]]] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.sequences = list(sequences)
        self.window_frames = int(window_frames)
        self.stride_frames = int(stride_frames)
        self.image_size = int(image_size)
        self.max_imu_samples = int(max_imu_samples)
        self.include_depth = bool(include_depth)
        if acc_source not in {"acc", "acc_nograv", "acc_nograv_body"}:
            raise ValueError("acc_source must be acc, acc_nograv, or acc_nograv_body")
        self.acc_source = acc_source
        self.cache_size = int(cache_size)
        self._metadata_cache: OrderedDict[str, dict] = OrderedDict()
        self.samples: list[tuple[str, int]] = []
        if samples is not None:
            allowed = set(self.sequences)
            for sequence, start in samples:
                if sequence not in allowed:
                    raise ValueError(f"explicit sample references unknown sequence {sequence}")
                if int(start) < 0:
                    raise ValueError("explicit sample start must be non-negative")
                self.samples.append((sequence, int(start)))
        else:
            for sequence in self.sequences:
                sequence_path = self._sequence_path(sequence)
                cam_time_path = sequence_path / "imu" / "cam_time.txt"
                with cam_time_path.open() as stream:
                    frames = sum(1 for line in stream if line.strip())
                for start in range(0, max(0, frames - self.window_frames + 1), self.stride_frames):
                    self.samples.append((sequence, start))
                if frames >= self.window_frames:
                    last = frames - self.window_frames
                    if not self.samples or self.samples[-1] != (sequence, last):
                        self.samples.append((sequence, last))
        if not self.samples:
            raise RuntimeError("no TartanAir windows were generated")

    def _sequence_path(self, sequence: str) -> Path:
        environment, difficulty, trajectory = sequence.split("/")
        return self.data_root / environment / f"Data_{difficulty}" / trajectory

    def _load_metadata(self, sequence: str) -> dict:
        if sequence in self._metadata_cache:
            value = self._metadata_cache.pop(sequence)
            self._metadata_cache[sequence] = value
            return value
        path = self._sequence_path(sequence)
        image_files = sorted((path / "image_lcam_front").glob("*.png"))
        depth_files = sorted((path / "depth_lcam_front").glob("*.png"))
        pose = np.loadtxt(path / "pose_lcam_front.txt", dtype=np.float64)
        imu_dir = path / "imu"
        imu_time = np.loadtxt(imu_dir / "imu_time.txt", dtype=np.float64).reshape(-1)
        cam_time = np.loadtxt(imu_dir / "cam_time.txt", dtype=np.float64).reshape(-1)
        acc = np.loadtxt(imu_dir / f"{self.acc_source}.txt", dtype=np.float32).reshape(-1, 3)
        gyro = np.loadtxt(imu_dir / "gyro.txt", dtype=np.float32).reshape(-1, 3)
        count = min(len(image_files), len(depth_files) if self.include_depth else 10**9, len(pose), len(cam_time))
        if count < self.window_frames:
            raise RuntimeError(f"sequence {sequence} has only {count} aligned frames")
        value = {
            "path": path,
            "image_files": image_files[:count],
            "depth_files": depth_files[:count] if self.include_depth else [],
            "pose": pose[:count],
            "cam_time": cam_time[:count],
            "imu_time": imu_time,
            "acc": acc[: len(imu_time)],
            "gyro": gyro[: len(imu_time)],
        }
        self._metadata_cache[sequence] = value
        while len(self._metadata_cache) > self.cache_size:
            self._metadata_cache.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self.samples)

    def _imu_interval(
        self, metadata: dict, start_time: float, end_time: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        imu_time = metadata["imu_time"]
        left = int(np.searchsorted(imu_time, start_time, side="left"))
        right = int(np.searchsorted(imu_time, end_time, side="right"))
        left = min(max(left, 0), len(imu_time) - 1)
        right = min(max(right, left + 1), len(imu_time))
        indices = np.arange(left, right, dtype=np.int64)
        if len(indices) > self.max_imu_samples:
            raise RuntimeError(
                f"TartanAir interval {start_time:.6f}->{end_time:.6f} has "
                f"{len(indices)} IMU samples, exceeding max_imu_samples="
                f"{self.max_imu_samples}; increase the capacity to keep every sample"
            )
        times = imu_time[indices]
        # Each selected measurement is held until the next measurement or the
        # RGB boundary.  Using only ``diff(times)`` drops the tail whenever
        # the final IMU timestamp is just before ``end_time`` (for example,
        # a 90 ms sum for a 100 ms RGB interval).  That creates a train/test
        # timing mismatch and biases preintegration.
        dt = np.diff(times, prepend=start_time)
        dt[-1] += max(float(end_time) - float(times[-1]), 0.0)
        dt = np.maximum(dt, 0.0)
        interval_duration = max(float(end_time) - float(start_time), 0.0)
        total_duration = float(dt.sum())
        if total_duration > 0.0:
            dt *= interval_duration / total_duration
        dt = dt.astype(np.float32)
        values = np.concatenate((metadata["acc"][indices], metadata["gyro"][indices]), axis=-1)
        return values.astype(np.float32), dt, times.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sequence, start = self.samples[index]
        metadata = self._load_metadata(sequence)
        stop = start + self.window_frames
        image_files = metadata["image_files"][start:stop]
        images = torch.stack([_load_rgb(path, self.image_size) for path in image_files])
        if self.include_depth:
            depth_files = metadata["depth_files"][start:stop]
            depth = torch.stack([_load_depth(path, self.image_size) for path in depth_files])
            depth_mask = torch.isfinite(depth) & (depth > 0.05) & (depth < 80.0)
        else:
            depth = torch.empty(0)
            depth_mask = torch.empty(0, dtype=torch.bool)

        cam_time = metadata["cam_time"][start:stop]
        # The first frame in a window still represents the interval ending at
        # its global timestamp.  Use the actual preceding camera timestamp
        # (or the measured cadence for the sequence's first frame) so the
        # camera duration and its IMU segment describe the same interval.
        if start == 0:
            first_interval = float(
                np.median(np.diff(metadata["cam_time"][: min(16, len(metadata["cam_time"]))]))
            )
            previous_camera_time = float(cam_time[0] - first_interval)
        else:
            previous_camera_time = float(metadata["cam_time"][start - 1])
        imu_segments: list[np.ndarray] = []
        imu_dts: list[np.ndarray] = []
        for local in range(self.window_frames):
            frame = start + local
            if frame == 0:
                interval = float(np.median(np.diff(metadata["cam_time"][: min(16, len(metadata["cam_time"]))])))
                begin = float(metadata["cam_time"][frame] - interval)
            else:
                begin = float(metadata["cam_time"][frame - 1])
            values, dt, _ = self._imu_interval(metadata, begin, float(metadata["cam_time"][frame]))
            imu_segments.append(values)
            imu_dts.append(dt)
        max_samples = max(len(values) for values in imu_segments)
        imu = torch.zeros(self.window_frames, max_samples, 6, dtype=torch.float32)
        imu_dt = torch.zeros(self.window_frames, max_samples, dtype=torch.float32)
        imu_mask = torch.zeros(self.window_frames, max_samples, dtype=torch.bool)
        for frame, (values, dt) in enumerate(zip(imu_segments, imu_dts)):
            count = len(values)
            imu[frame, :count] = torch.from_numpy(values)
            imu_dt[frame, :count] = torch.from_numpy(dt)
            imu_mask[frame, :count] = True

        poses = _pose_to_matrix(metadata["pose"][start:stop])
        dt_camera = np.diff(cam_time, prepend=previous_camera_time).astype(np.float32)
        scale = self.image_size / 640.0
        intrinsics = torch.tensor(
            (
                (320.0 * scale, 0.0, 320.0 * scale),
                (0.0, 320.0 * scale, 320.0 * scale),
                (0.0, 0.0, 1.0),
            ),
            dtype=torch.float32,
        ).expand(self.window_frames, -1, -1).clone()
        return {
            "images": images,
            "depth": depth,
            "depth_mask": depth_mask,
            "poses": poses,
            "gt_intrinsics": intrinsics,
            "camera_times": torch.from_numpy(cam_time.astype(np.float32)),
            "camera_dt": torch.from_numpy(dt_camera),
            "imu": imu,
            "imu_dt": imu_dt,
            "imu_mask": imu_mask,
            "gravity_camera": torch.tensor((0.0, 0.0, -9.80665), dtype=torch.float32),
            "sequence": sequence,
            "start_frame": torch.tensor(start, dtype=torch.long),
        }
