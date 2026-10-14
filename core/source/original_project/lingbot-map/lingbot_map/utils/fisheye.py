"""Calibrated equidistant-fisheye preprocessing for pinhole geometry models."""

from __future__ import annotations

import csv
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


def load_equidistant_calibration(path: str | Path, camera: str = "cam0") -> dict:
    """Load one camera entry from a Kalibr/TUM-VI camchain YAML file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on optional use
        raise RuntimeError(
            "Fisheye rectification requires PyYAML; install it before using "
            "--fisheye_calibration."
        ) from exc
    with Path(path).open(encoding="utf-8") as handle:
        calibration = yaml.safe_load(handle)[camera]
    if calibration.get("camera_model") != "pinhole":
        raise ValueError(f"Unsupported camera model: {calibration.get('camera_model')}")
    if calibration.get("distortion_model") != "equidistant":
        raise ValueError(
            "This preprocessing expects OpenCV/Kalibr equidistant distortion, "
            f"got {calibration.get('distortion_model')}"
        )
    return calibration


def rectify_equidistant_images(
    paths: list[str | Path],
    calibration: dict,
    target_size: int = 518,
    fov_degrees: float = 90.0,
    workers: int = 16,
) -> torch.Tensor:
    """Rectify calibrated fisheye images and return ``[N,3,H,W]`` RGB tensors.

    The LingBot camera head decodes pinhole intrinsics. A TUM-VI equidistant
    image therefore needs to be rectified before inference; the operation is
    deterministic and does not use a checkpoint or ground truth.
    """
    if not paths:
        raise ValueError("At least one image path is required")
    if not 30.0 <= fov_degrees <= 150.0:
        raise ValueError("fov_degrees must be in [30, 150]")
    width, height = map(int, calibration["resolution"])
    fx, fy, cx, cy = map(float, calibration["intrinsics"])
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    distortion = np.asarray(calibration["distortion_coeffs"], dtype=np.float64)
    focal = 0.5 * min(width, height) / math.tan(math.radians(fov_degrees) * 0.5)
    rectified_matrix = np.array(
        [
            [focal, 0.0, (width - 1) * 0.5],
            [0.0, focal, (height - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        np.eye(3, dtype=np.float64),
        rectified_matrix,
        (width, height),
        cv2.CV_32FC1,
    )

    def load_one(path_value: str | Path) -> torch.Tensor:
        image = cv2.imread(str(path_value), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path_value)
        image = cv2.remap(
            image, map_x, map_y, interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
        )
        image = cv2.resize(
            image, (target_size, target_size), interpolation=cv2.INTER_CUBIC
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)

    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(paths))) as pool:
        images = list(
            tqdm(
                pool.map(load_one, paths),
                total=len(paths),
                desc="Rectifying images",
            )
        )
    return torch.stack(images)


def tumvi_image_paths(
    sequence_root: str | Path,
    camera: str = "cam0",
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> tuple[list[Path], np.ndarray]:
    """Read TUM-VI EuRoC image CSV and return selected paths/timestamps."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    csv_path = Path(sequence_root) / "mav0" / camera / "data.csv"
    paths: list[Path] = []
    timestamps: list[int] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(line for line in handle if not line.startswith("#"))
        for row in rows:
            if row:
                timestamps.append(int(row[0]))
                paths.append(csv_path.parent / "data" / row[1].strip())
    selected = np.arange(0, len(paths), frame_stride, dtype=np.int64)
    if max_frames is not None:
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2")
        selected = selected[:max_frames]
    return [paths[index] for index in selected], np.asarray(timestamps)[selected]
