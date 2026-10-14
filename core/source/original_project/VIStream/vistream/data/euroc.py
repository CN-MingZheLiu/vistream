"""EuRoC-format and converted KAIST sequence readers for evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import yaml

from ..geometry.lie import invert_se3, quaternion_xyzw_to_matrix


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as stream:
        return [row for row in csv.reader(stream) if row and not row[0].startswith("#")]


def _read_matrix_yaml(path: Path, key: str) -> Optional[torch.Tensor]:
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    node = data.get(key, {})
    values = node.get("data") if isinstance(node, dict) else None
    if values is None:
        return None
    rows = int(node.get("rows", 4))
    cols = int(node.get("cols", 4))
    return torch.tensor(values, dtype=torch.float32).reshape(rows, cols)


def _load_image(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[float, float, float, float]]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = max(width, height)
    scale = image_size / side
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    array = np.array(resized, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    pad_h = image_size - tensor.shape[-2]
    pad_w = image_size - tensor.shape[-1]
    top, left = max(0, pad_h // 2), max(0, pad_w // 2)
    tensor = F.pad(tensor, (left, pad_w - left, top, pad_h - top), value=0.0)
    return tensor, (scale, left, top, float(side))


def _pose_from_row(row: list[str]) -> torch.Tensor:
    values = np.asarray([float(value) for value in row[1:8]], dtype=np.float32)
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = quaternion_xyzw_to_matrix(torch.from_numpy(values[3:7]))
    transform[:3, 3] = torch.from_numpy(values[:3])
    return transform


class EurocWindowDataset(Dataset):
    """Timestamp-aligned windows from a EuRoC or converted KAIST sequence."""

    def __init__(
        self,
        root: str | Path,
        sequence: str,
        image_size: int = 518,
        window_frames: int = 16,
        stride_frames: int = 8,
        max_imu_samples: int = 64,
        camera_yaml: str | Path | None = None,
        camera_from_imu: torch.Tensor | None = None,
    ) -> None:
        self.root = Path(root)
        self.sequence = sequence
        self.sequence_path = self.root / sequence / "mav0"
        if not self.sequence_path.is_dir():
            raise FileNotFoundError(f"EuRoC sequence not found: {self.sequence_path}")
        self.image_size = image_size
        self.window_frames = window_frames
        self.max_imu_samples = max_imu_samples
        camera_rows = _rows(self.sequence_path / "cam0" / "data.csv")
        self.image_times = np.asarray([int(row[0]) * 1e-9 for row in camera_rows], dtype=np.float64)
        self.image_files = [self.sequence_path / "cam0" / "data" / row[1] for row in camera_rows]
        imu_rows = _rows(self.sequence_path / "imu0" / "data.csv")
        imu_values = np.asarray([[float(value) for value in row[1:]] for row in imu_rows], dtype=np.float32)
        self.imu_times = np.asarray([int(row[0]) * 1e-9 for row in imu_rows], dtype=np.float64)
        self.imu = np.concatenate((imu_values[:, 3:6], imu_values[:, :3]), axis=-1)
        gt_path = self.sequence_path / "state_groundtruth_estimate0" / "data.csv"
        gt_rows = _rows(gt_path)
        gt_times = np.asarray([int(row[0]) * 1e-9 for row in gt_rows], dtype=np.float64)
        gt_values = np.asarray([[float(value) for value in row[1:8]] for row in gt_rows], dtype=np.float32)
        # Position interpolation and normalized-linear quaternion interpolation.
        indices = np.searchsorted(gt_times, self.image_times).clip(1, len(gt_times) - 1)
        alpha = ((self.image_times - gt_times[indices - 1]) / (gt_times[indices] - gt_times[indices - 1] + 1e-12)).astype(np.float32)
        position = (1 - alpha[:, None]) * gt_values[indices - 1, :3] + alpha[:, None] * gt_values[indices, :3]
        # EuRoC ground truth stores quaternions as WXYZ; geometry utilities use XYZW.
        q0 = np.concatenate((gt_values[indices - 1, 4:7], gt_values[indices - 1, 3:4]), axis=-1)
        q1 = np.concatenate((gt_values[indices, 4:7], gt_values[indices, 3:4]), axis=-1)
        signs = np.sign((q0 * q1).sum(axis=-1, keepdims=True))
        quaternion = (q0 + alpha[:, None] * (q1 * signs - q0))
        quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(1e-8)
        self.poses = torch.eye(4, dtype=torch.float32).repeat(len(position), 1, 1)
        self.poses[:, :3, :3] = quaternion_xyzw_to_matrix(torch.from_numpy(quaternion))
        self.poses[:, :3, 3] = torch.from_numpy(position)

        camera_yaml_path = Path(camera_yaml) if camera_yaml else self.sequence_path / "cam0" / "sensor.yaml"
        cfg = yaml.safe_load(camera_yaml_path.read_text()) if camera_yaml_path.is_file() else {}
        camera_cfg = (cfg or {}).get("cam0", cfg or {})
        intrinsics = camera_cfg.get("intrinsics", [1.0, 1.0, 0.0, 0.0])
        with Image.open(self.image_files[0]) as image:
            width, height = image.size
        side = max(width, height)
        scale = image_size / side
        left = (image_size - round(width * scale)) / 2.0
        top = (image_size - round(height * scale)) / 2.0
        self.intrinsics = torch.tensor(
            ((intrinsics[0] * scale, 0, intrinsics[2] * scale + left),
             (0, intrinsics[1] * scale, intrinsics[3] * scale + top),
             (0, 0, 1)), dtype=torch.float32
        ).expand(len(self.image_files), -1, -1).clone()
        camera_to_body = _read_matrix_yaml(camera_yaml_path, "T_BS")
        yaml_camera_from_imu = None
        if "T_cam_imu" in camera_cfg:
            yaml_camera_from_imu = torch.tensor(
                camera_cfg["T_cam_imu"], dtype=torch.float32
            )
            camera_to_body = invert_se3(yaml_camera_from_imu)
        if camera_to_body is not None:
            # Ground truth is the body/IMU pose; LingBot predicts the camera pose.
            self.poses = self.poses @ camera_to_body.unsqueeze(0)
        if camera_from_imu is not None:
            self.camera_from_imu = camera_from_imu.float()
        elif yaml_camera_from_imu is not None:
            self.camera_from_imu = yaml_camera_from_imu
        else:
            self.camera_from_imu = invert_se3(camera_to_body) if camera_to_body is not None else torch.eye(4)
        gravity_imu = torch.from_numpy(self.imu[: min(200, len(self.imu)), :3].mean(axis=0))
        self.gravity_camera = self.camera_from_imu[:3, :3] @ gravity_imu
        self.samples = [
            start for start in range(0, max(0, len(self.image_files) - window_frames + 1), stride_frames)
        ]
        last = len(self.image_files) - window_frames
        if last >= 0 and (not self.samples or self.samples[-1] != last):
            self.samples.append(last)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        start = self.samples[index]
        stop = start + self.window_frames
        images = []
        for path in self.image_files[start:stop]:
            image, _ = _load_image(path, self.image_size)
            images.append(image)
        images_tensor = torch.stack(images)
        imu_segments, imu_dts = [], []
        for frame in range(start, stop):
            duration = self.image_times[1] - self.image_times[0] if frame == 0 else self.image_times[frame] - self.image_times[frame - 1]
            begin = self.image_times[frame] - duration if frame == 0 else self.image_times[frame - 1]
            left = int(np.searchsorted(self.imu_times, begin, side="left"))
            right = int(np.searchsorted(self.imu_times, self.image_times[frame], side="right"))
            left = min(max(left, 0), len(self.imu_times) - 1)
            right = min(max(right, left + 1), len(self.imu_times))
            ids = np.arange(left, right)
            if len(ids) > self.max_imu_samples:
                raise RuntimeError(
                    f"EuRoC {self.sequence} frame {frame} has {len(ids)} IMU "
                    f"samples, exceeding max_imu_samples={self.max_imu_samples}; "
                    "increase the capacity to keep every sample"
                )
            times = self.imu_times[ids]
            imu_segments.append(self.imu[ids])
            imu_dts.append(np.clip(np.diff(times, prepend=begin), 1e-5, None).astype(np.float32))
        max_samples = max(map(len, imu_segments))
        imu = torch.zeros(self.window_frames, max_samples, 6)
        imu_dt = torch.zeros(self.window_frames, max_samples)
        imu_mask = torch.zeros(self.window_frames, max_samples, dtype=torch.bool)
        for frame, (values, dts) in enumerate(zip(imu_segments, imu_dts)):
            count = len(values)
            imu[frame, :count] = torch.from_numpy(values)
            imu_dt[frame, :count] = torch.from_numpy(dts)
            imu_mask[frame, :count] = True
        absolute_camera_times = self.image_times[start:stop]
        camera_times = torch.from_numpy(
            (absolute_camera_times - absolute_camera_times[0]).astype(np.float32)
        )
        first_dt = absolute_camera_times[1] - absolute_camera_times[0]
        camera_dt = torch.from_numpy(
            np.diff(absolute_camera_times, prepend=absolute_camera_times[0] - first_dt).astype(np.float32)
        )
        return {
            "images": images_tensor,
            "depth": torch.empty(0),
            "depth_mask": torch.empty(0, dtype=torch.bool),
            "poses": self.poses[start:stop].clone(),
            "gt_intrinsics": self.intrinsics[start:stop].clone(),
            "camera_times": camera_times,
            "camera_dt": camera_dt,
            "imu": imu,
            "imu_dt": imu_dt,
            "imu_mask": imu_mask,
            "rotation_camera_from_imu": self.camera_from_imu[:3, :3].clone(),
            "gravity_camera": self.gravity_camera.clone(),
            "sequence": self.sequence,
            "start_frame": torch.tensor(start, dtype=torch.long),
        }
