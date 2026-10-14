"""KAIST VIO adapter; bags must first be converted to EuRoC CSV layout."""

from __future__ import annotations

from pathlib import Path

from .euroc import EurocWindowDataset


KAIST_SEQUENCE_NAMES = (
    "circle", "circle_fast", "circle_head",
    "infinite", "infinite_fast", "infinite_head",
    "rotation", "rotation_fast", "square", "square_fast", "square_head",
)


def kaist_dataset(
    converted_root: str | Path,
    sequence: str,
    camera_yaml: str | Path = "/data1/mingzheliu/datasets/KAIST_VIO/kaistviodataset/config/cam-imu.yaml",
    **kwargs,
) -> EurocWindowDataset:
    """Build a window dataset from output of ``convert_kaist_to_euroc.py``."""
    if sequence not in KAIST_SEQUENCE_NAMES:
        raise ValueError(f"unknown KAIST sequence {sequence!r}")
    return EurocWindowDataset(
        converted_root,
        sequence,
        camera_yaml=camera_yaml,
        **kwargs,
    )

