"""Pluggable IMU preprocessing protocol.

Processors may wrap a learned correction model such as AirIMU, a calibrated
classical filter, or a precomputed external estimator. The fusion backend only
consumes this normalized SI-unit contract and never inspects processor internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class IMUProcessingContext:
    frame_timestamps: np.ndarray
    rotation_camera_from_imu: np.ndarray = field(default_factory=lambda: np.eye(3))
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IMUStream:
    """Normalized processor output.

    ``samples`` uses ``[timestamp_s, ax, ay, az, gx, gy, gz]``. A processor
    that already performs preintegration can instead populate
    ``interval_delta_rotations`` with one SO(3) matrix per camera interval.
    """

    samples: np.ndarray | None = None
    interval_delta_rotations: np.ndarray | None = None
    interval_confidence: np.ndarray | None = None
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class IMUProcessor(Protocol):
    name: str

    def process(self, samples: np.ndarray, context: IMUProcessingContext) -> IMUStream:
        """Return corrected SI-unit samples or camera-interval rotations."""


def _validated_samples(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(
            "IMU samples must have shape [N,7] = timestamp, acc xyz, gyro xyz"
        )
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("IMU samples must contain at least two finite rows")
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    values = values[np.r_[True, np.diff(values[:, 0]) > 0.0]]
    if len(values) < 2:
        raise ValueError("IMU timestamps must contain at least two distinct values")
    return values


class RawIMUProcessor:
    """Identity processor for calibrated raw IMU samples."""

    name = "raw"

    def process(self, samples: np.ndarray, context: IMUProcessingContext) -> IMUStream:
        del context
        return IMUStream(samples=_validated_samples(samples), source=self.name)


class NPZIMUProcessor:
    """Load output produced by any external IMU processing pipeline.

    Supported archives contain either ``samples`` in the normalized contract,
    ``timestamps`` plus ``acc`` and ``gyro``, or ``delta_rotations``. Optional
    confidence is read from ``confidence``.
    """

    name = "npz"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def process(self, samples: np.ndarray, context: IMUProcessingContext) -> IMUStream:
        del samples
        with np.load(self.path, allow_pickle=False) as archive:
            keys = set(archive.files)
            confidence = (
                np.asarray(archive["confidence"], dtype=np.float64)
                if "confidence" in keys
                else None
            )
            if "samples" in keys:
                return IMUStream(
                    samples=_validated_samples(archive["samples"]),
                    interval_confidence=confidence,
                    source=f"npz:{self.path}",
                )
            if {"timestamps", "acc", "gyro"} <= keys:
                normalized = np.column_stack(
                    (archive["timestamps"], archive["acc"], archive["gyro"])
                )
                return IMUStream(
                    samples=_validated_samples(normalized),
                    interval_confidence=confidence,
                    source=f"npz:{self.path}",
                )
            if "delta_rotations" in keys:
                rotations = np.asarray(archive["delta_rotations"], dtype=np.float64)
                expected = len(context.frame_timestamps) - 1
                if rotations.shape != (expected, 3, 3):
                    raise ValueError(
                        f"delta_rotations must have shape {(expected, 3, 3)}, "
                        f"got {rotations.shape}"
                    )
                return IMUStream(
                    interval_delta_rotations=rotations,
                    interval_confidence=confidence,
                    source=f"npz:{self.path}",
                )
        raise ValueError(
            f"{self.path} must contain samples, timestamps+acc+gyro, or delta_rotations"
        )


def load_processor(specification: str) -> IMUProcessor:
    """Load ``raw``, ``npz:/path``, or ``module:factory`` processor specs."""
    if specification == "raw":
        return RawIMUProcessor()
    if specification.startswith("npz:"):
        return NPZIMUProcessor(specification[4:])
    if ":" not in specification:
        raise ValueError(
            "processor must be 'raw', 'npz:/path', or 'python.module:factory'"
        )
    module_name, factory_name = specification.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    processor = factory()
    if not isinstance(processor, IMUProcessor):
        raise TypeError(f"{specification} did not create an IMUProcessor")
    return processor
