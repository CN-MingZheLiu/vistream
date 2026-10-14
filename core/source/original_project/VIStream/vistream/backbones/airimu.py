"""Frozen, checkpoint-compatible AirIMU CodeNetKITTI feature backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class _CNNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (3, 32, 64, 128)
        kernels = (7, 3, 3)
        padding = (3, 1, 1)
        layers: list[nn.Module] = []
        for index in range(3):
            layers.extend(
                (
                    nn.Conv1d(
                        channels[index], channels[index + 1], kernels[index],
                        padding=padding[index],
                    ),
                    nn.BatchNorm1d(channels[index + 1]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def _decoder() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3)
    )


def _covariance_decoder() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 3)
    )


class FrozenAirIMUKITTI(nn.Module):
    """Exact trainable-layer layout of AirIMU's ``CodeNetKITTI``.

    The covariance decoders are loaded for checkpoint integrity but are never
    used as uncertainty estimates: the KITTI model was trained with
    ``propcov=false``.  ``forward_causal`` executes the frozen CNN on prefixes
    and takes only the newest feature, preventing convolutional future leakage.
    """

    feature_dim = 256

    def __init__(self) -> None:
        super().__init__()
        self.accEncoder = _CNNEncoder()
        self.gyroEncoder = _CNNEncoder()
        self.accDecoder = _decoder()
        self.gyroDecoder = _decoder()
        self.accCovDecoder = _covariance_decoder()
        self.gyroCovDecoder = _covariance_decoder()
        self.register_buffer("gyro_std", torch.ones(1, 3))
        self.register_buffer("acc_std", torch.ones(1, 3))
        self._checkpoint_path: str | None = None
        self._loaded_keys = 0

    @staticmethod
    def _state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise TypeError("AirIMU checkpoint must contain a state dictionary")
        state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        if not isinstance(state, dict):
            raise TypeError("AirIMU checkpoint state is not a dictionary")
        clean: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            for prefix in ("module.", "imu_encoder.", "airimu."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            clean[key] = value
        return clean

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "FrozenAirIMUKITTI":
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"AirIMU checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        state = cls._state_dict(checkpoint)
        model = cls()
        current = model.state_dict()
        compatible = {
            key: value for key, value in state.items()
            if key in current and current[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        required_prefixes = ("accEncoder.", "gyroEncoder.", "accDecoder.", "gyroDecoder.")
        missing_required = [
            key for key in missing if key.startswith(required_prefixes)
        ]
        if missing_required:
            raise RuntimeError(
                "AirIMU checkpoint is incompatible; missing required tensors: "
                + ", ".join(missing_required[:8])
            )
        source_required = [key for key in state if key.startswith(required_prefixes)]
        if len(compatible) < len(source_required):
            raise RuntimeError("not all required AirIMU tensors matched the model")
        if unexpected:
            raise RuntimeError(f"unexpected AirIMU tensors after filtering: {unexpected}")
        model._checkpoint_path = str(path.resolve())
        model._loaded_keys = len(compatible)
        model.freeze()
        return model

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenAirIMUKITTI":
        super().train(False)
        return self

    def _encode(self, acc: torch.Tensor, gyro: torch.Tensor) -> tuple[torch.Tensor, ...]:
        acc_feature = self.accEncoder(acc.transpose(1, 2)).transpose(1, 2)
        gyro_feature = self.gyroEncoder(gyro.transpose(1, 2)).transpose(1, 2)
        return acc_feature, gyro_feature

    @torch.no_grad()
    def forward_causal(self, acc: torch.Tensor, gyro: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return one AirIMU output per RGB time using only that time's prefix."""
        if acc.shape != gyro.shape or acc.ndim != 3 or acc.shape[-1] != 3:
            raise ValueError("acc and gyro must both have shape [B,S,3]")
        features: list[torch.Tensor] = []
        acc_corrections: list[torch.Tensor] = []
        gyro_corrections: list[torch.Tensor] = []
        for frame in range(acc.shape[1]):
            acc_feature, gyro_feature = self._encode(
                acc[:, : frame + 1], gyro[:, : frame + 1]
            )
            newest_acc = acc_feature[:, -1]
            newest_gyro = gyro_feature[:, -1]
            features.append(torch.cat((newest_acc, newest_gyro), dim=-1))
            acc_corrections.append(self.accDecoder(newest_acc))
            gyro_corrections.append(self.gyroDecoder(newest_gyro))
        return {
            "features": torch.stack(features, dim=1),
            "acc_correction": torch.stack(acc_corrections, dim=1),
            "gyro_correction": torch.stack(gyro_corrections, dim=1),
        }

