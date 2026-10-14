"""YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .losses import LossWeights
from .model.adapter import FusionAdapterConfig
from .model.system import VIStreamConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return data


def model_config(data: dict[str, Any]) -> VIStreamConfig:
    """Build a model config; relative backbone paths are package-local."""
    values = dict(data["model"])
    # Configs written before pose fusion existed must continue to load their
    # token/AirIMU checkpoints. New configs opt into the safer path explicitly.
    if "fusion_mode" not in values and "imu_architecture" not in values:
        values["fusion_mode"] = "token"
        values["imu_architecture"] = "airimu"
    values["adapter"] = FusionAdapterConfig(**values.get("adapter", {}))
    return VIStreamConfig(**values)


def loss_weights(data: dict[str, Any]) -> LossWeights:
    return LossWeights(**data.get("loss", {}))
