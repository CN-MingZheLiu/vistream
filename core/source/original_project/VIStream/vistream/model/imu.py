"""Dual-rate, RGB-causal IMU tokenization and SE(3) preintegration."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..backbones.airimu import FrozenAirIMUKITTI
from ..geometry.lie import se3_log, so3_exp


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights * mask.to(weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (values * weights.unsqueeze(-1)).sum(dim=-2)


class RawIntervalTokenizer(nn.Module):
    """Encode all 100 Hz samples in each completed RGB interval."""

    def __init__(self, output_dim: int = 192, sample_dim: int = 128) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.sample_encoder = nn.Sequential(
            nn.Linear(8, sample_dim),
            nn.LayerNorm(sample_dim),
            nn.SiLU(),
            nn.Linear(sample_dim, sample_dim),
            nn.SiLU(),
        )
        self.score = nn.Linear(sample_dim, 1)
        self.frame_gru = nn.GRU(sample_dim, output_dim, batch_first=True)

    def forward(
        self,
        corrected_acc: torch.Tensor,
        corrected_gyro: torch.Tensor,
        dt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        elapsed = torch.cumsum(dt * mask.to(dt.dtype), dim=-1)
        duration = elapsed[..., -1:].clamp_min(1e-6)
        relative_time = elapsed / duration
        samples = torch.cat(
            (
                corrected_acc,
                corrected_gyro,
                dt.unsqueeze(-1),
                relative_time.unsqueeze(-1),
            ),
            dim=-1,
        )
        encoded = self.sample_encoder(samples)
        scores = self.score(encoded).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * mask.to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = (encoded * attention.unsqueeze(-1)).sum(dim=-2)
        causal_frames, _ = self.frame_gru(pooled)
        return causal_frames


class CausalPreintegrator(nn.Module):
    """Preintegrate completed IMU intervals into local SE(3) increments."""

    def forward(
        self,
        acc: torch.Tensor,
        gyro: torch.Tensor,
        dt: torch.Tensor,
        mask: torch.Tensor,
        gravity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if gravity.ndim == 1:
            gravity = gravity.unsqueeze(0).expand(acc.shape[0], -1)
        batch, frames, samples = dt.shape
        eye = torch.eye(3, device=acc.device, dtype=acc.dtype)
        rotations: list[torch.Tensor] = []
        translations: list[torch.Tensor] = []
        velocities: list[torch.Tensor] = []
        for frame in range(frames):
            rotation = eye.expand(batch, 3, 3).clone()
            velocity = torch.zeros(batch, 3, device=acc.device, dtype=acc.dtype)
            position = torch.zeros_like(velocity)
            for sample in range(samples):
                valid = mask[:, frame, sample].to(acc.dtype).unsqueeze(-1)
                delta_t = dt[:, frame, sample].unsqueeze(-1) * valid
                acceleration = acc[:, frame, sample] - gravity
                acceleration_local = (rotation @ acceleration.unsqueeze(-1)).squeeze(-1)
                position = position + velocity * delta_t + 0.5 * acceleration_local * delta_t.square()
                velocity = velocity + acceleration_local * delta_t
                increment = so3_exp(gyro[:, frame, sample] * delta_t)
                candidate = rotation @ increment
                rotation = torch.where(valid.unsqueeze(-1).bool(), candidate, rotation)
            rotations.append(rotation)
            translations.append(position)
            velocities.append(velocity)
        rotation_tensor = torch.stack(rotations, dim=1)
        translation_tensor = torch.stack(translations, dim=1)
        transform = torch.zeros(
            batch, frames, 4, 4, device=acc.device, dtype=acc.dtype
        )
        transform[..., :3, :3] = rotation_tensor
        transform[..., :3, 3] = translation_tensor
        transform[..., 3, 3] = 1.0
        return {
            "delta_transform": transform,
            "delta_twist": se3_log(transform),
            "delta_velocity": torch.stack(velocities, dim=1),
        }


class DualRateIMUEncoder(nn.Module):
    """Fuse frozen 10 Hz AirIMU features with trainable raw 100 Hz motion."""

    stats_dim = 10

    def __init__(
        self,
        airimu: FrozenAirIMUKITTI,
        raw_feature_dim: int = 192,
    ) -> None:
        super().__init__()
        self.airimu = airimu
        self.raw_tokenizer = RawIntervalTokenizer(output_dim=raw_feature_dim)
        self.preintegrator = CausalPreintegrator()
        self.raw_feature_dim = raw_feature_dim
        # AirIMU was trained on KITTI and its correction is not guaranteed to
        # transfer to TartanAir/EuRoC.  Start almost at the raw-IMU identity
        # and learn bounded per-frame correction gates instead of adding the
        # frozen correction unconditionally.
        self.correction_gate = nn.Sequential(
            nn.LayerNorm(262),
            nn.Linear(262, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        nn.init.constant_(self.correction_gate[-1].bias, -6.0)
        # A causal history state carries slowly varying accelerometer/gyro
        # bias across RGB intervals.  The head starts at zero, so it cannot
        # disturb the frozen AirIMU/KITTI behavior at initialization.
        self.bias_gru = nn.GRU(262, 64, batch_first=True)
        self.bias_head = nn.Linear(64, 6)
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

    @staticmethod
    def _rotate(values: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        if rotation.ndim == 2:
            rotation = rotation.unsqueeze(0).expand(values.shape[0], -1, -1)
        return torch.einsum("bij,bsmj->bsmi", rotation, values)

    def forward(
        self,
        imu: torch.Tensor,
        imu_dt: torch.Tensor,
        imu_mask: torch.Tensor,
        rotation_camera_from_imu: Optional[torch.Tensor] = None,
        gravity_camera: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if imu.ndim != 4 or imu.shape[-1] != 6:
            raise ValueError("imu must have shape [B,S,M,6] with [acc,gyro]")
        if imu_dt.shape != imu.shape[:-1] or imu_mask.shape != imu.shape[:-1]:
            raise ValueError("imu_dt and imu_mask must match imu's [B,S,M] dimensions")
        acc, gyro = imu.split(3, dim=-1)
        weights = imu_dt.clamp_min(0.0)
        acc_rgb = _masked_mean(acc, imu_mask, weights)
        gyro_rgb = _masked_mean(gyro, imu_mask, weights)

        # CodeNetKITTI is frozen and evaluated on causal RGB-rate prefixes.
        air = self.airimu.forward_causal(acc_rgb.float(), gyro_rgb.float())
        correction_context = torch.cat(
            (air["features"].float(), acc_rgb.float(), gyro_rgb.float()), dim=-1
        )
        correction_gate = torch.sigmoid(self.correction_gate(correction_context))
        corrected_acc = acc + correction_gate[..., :1].unsqueeze(-2) * air["acc_correction"].unsqueeze(-2)
        corrected_gyro = gyro + correction_gate[..., 1:].unsqueeze(-2) * air["gyro_correction"].unsqueeze(-2)
        bias_state, _ = self.bias_gru(correction_context)
        bias = self.bias_head(bias_state)
        bias_scale = bias.new_tensor((0.5, 0.5, 0.5, 0.05, 0.05, 0.05))
        bias = torch.tanh(bias) * bias_scale
        corrected_acc = corrected_acc - bias[..., :3].unsqueeze(-2)
        corrected_gyro = corrected_gyro - bias[..., 3:].unsqueeze(-2)

        if rotation_camera_from_imu is None:
            rotation_camera_from_imu = torch.eye(
                3, device=imu.device, dtype=imu.dtype
            )
        rotation_camera_from_imu = rotation_camera_from_imu.to(imu)
        corrected_acc = self._rotate(corrected_acc, rotation_camera_from_imu)
        corrected_gyro = self._rotate(corrected_gyro, rotation_camera_from_imu)

        if gravity_camera is None:
            gravity_camera = torch.tensor(
                (0.0, 0.0, -9.80665), device=imu.device, dtype=imu.dtype
            )
        gravity_camera = gravity_camera.to(imu)
        raw_feature = self.raw_tokenizer(
            corrected_acc, corrected_gyro, imu_dt, imu_mask
        )
        preintegrated = self.preintegrator(
            corrected_acc.float(),
            corrected_gyro.float(),
            imu_dt.float(),
            imu_mask,
            gravity_camera.float(),
        )

        valid = imu_mask.to(imu.dtype)
        count = valid.sum(dim=-1).clamp_min(1.0)
        acc_norm = corrected_acc.norm(dim=-1)
        gyro_norm = corrected_gyro.norm(dim=-1)
        mean_acc = (acc_norm * valid).sum(dim=-1) / count
        mean_gyro = (gyro_norm * valid).sum(dim=-1) / count
        acc_variance = ((acc_norm - mean_acc.unsqueeze(-1)).square() * valid).sum(dim=-1) / count
        gyro_variance = ((gyro_norm - mean_gyro.unsqueeze(-1)).square() * valid).sum(dim=-1) / count
        duration = (imu_dt * valid).sum(dim=-1)
        max_gap = (imu_dt * valid).amax(dim=-1)
        stats = torch.stack(
            (
                mean_acc,
                mean_gyro,
                torch.sqrt(acc_variance + 1e-8),
                torch.sqrt(gyro_variance + 1e-8),
                duration,
                max_gap,
                air["acc_correction"].norm(dim=-1),
                air["gyro_correction"].norm(dim=-1),
                preintegrated["delta_twist"][..., :3].norm(dim=-1),
                preintegrated["delta_twist"][..., 3:].norm(dim=-1),
            ),
            dim=-1,
        )
        return {
            "airimu_feature": air["features"],
            "airimu_correction_gate": correction_gate,
            "bias_state": bias_state,
            "estimated_bias": bias,
            "raw_feature": raw_feature,
            "motion_stats": stats,
            "corrected_acc": corrected_acc,
            "corrected_gyro": corrected_gyro,
            **preintegrated,
        }


class CausalGRUIMUEncoder(nn.Module):
    """Strictly causal IMU encoder trained with the VIStream adapter.

    AirIMU's released KITTI CNN uses symmetric temporal padding. Running that
    network on every prefix changes the boundary distribution and repeatedly
    recomputes old frames. This encoder instead pools completed high-rate IMU
    intervals and carries one recurrent state at the RGB rate. Metric
    translation is deliberately left to visual-inertial pose fusion, because
    six-axis IMU cannot observe an unknown initial velocity by itself.
    """

    stats_dim = 10

    def __init__(
        self,
        raw_feature_dim: int = 192,
        sample_dim: int = 96,
        state_dim: int = 192,
    ) -> None:
        super().__init__()
        self.raw_feature_dim = raw_feature_dim
        self.sample_encoder = nn.Sequential(
            nn.Linear(8, sample_dim),
            nn.LayerNorm(sample_dim),
            nn.SiLU(),
            nn.Linear(sample_dim, sample_dim),
            nn.SiLU(),
        )
        # Preserve within-interval order. A weighted mean alone can describe
        # average motion but cannot distinguish, for example, a turn followed
        # by braking from braking followed by a turn. GRUCell updates are
        # explicitly masked so dropped/padded samples never alter the state.
        self.sample_gru = nn.GRUCell(sample_dim, sample_dim)
        self.interval_projection = nn.Sequential(
            nn.LayerNorm(sample_dim * 2),
            nn.Linear(sample_dim * 2, state_dim),
            nn.SiLU(),
        )
        self.sensor_projection = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, state_dim),
            nn.SiLU(),
        )
        self.temporal_gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.raw_head = nn.Linear(state_dim, raw_feature_dim)
        self.context_head = nn.Linear(state_dim, 256)
        self.bias_head = nn.Linear(state_dim, 6)
        self.preintegrator = CausalPreintegrator()

        # Identity IMU calibration at initialization. The downstream pose
        # fusion head is also zero initialized, preserving native LingBot.
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

    @staticmethod
    def _rotate(values: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        if rotation.ndim == 2:
            rotation = rotation.unsqueeze(0).expand(values.shape[0], -1, -1)
        return torch.einsum("bij,bsmj->bsmi", rotation, values)

    def forward(
        self,
        imu: torch.Tensor,
        imu_dt: torch.Tensor,
        imu_mask: torch.Tensor,
        rotation_camera_from_imu: Optional[torch.Tensor] = None,
        gravity_camera: Optional[torch.Tensor] = None,
        initial_hidden: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if imu.ndim != 4 or imu.shape[-1] != 6:
            raise ValueError("imu must have shape [B,S,M,6] with [acc,gyro]")
        if imu_dt.shape != imu.shape[:-1] or imu_mask.shape != imu.shape[:-1]:
            raise ValueError("imu_dt and imu_mask must match imu's [B,S,M] dimensions")

        acc, gyro = imu.split(3, dim=-1)
        valid = imu_mask.to(imu.dtype)
        elapsed = torch.cumsum(imu_dt * valid, dim=-1)
        duration = elapsed[..., -1:].clamp_min(1e-6)
        # Normalize physical channels before learning. The preintegrator below
        # still receives SI units; this normalization only removes sensor
        # scale differences from the neural input distribution.
        samples = torch.cat(
            (
                acc / 9.80665,
                gyro,
                imu_dt.unsqueeze(-1) / 0.01,
                (elapsed / duration).unsqueeze(-1),
            ),
            dim=-1,
        )
        encoded_samples = self.sample_encoder(samples)
        pooled = _masked_mean(
            encoded_samples, imu_mask, imu_dt.clamp_min(0.0)
        )
        sample_state = encoded_samples.new_zeros(
            encoded_samples.shape[0], encoded_samples.shape[1], encoded_samples.shape[-1]
        )
        for sample in range(encoded_samples.shape[2]):
            candidate = self.sample_gru(
                encoded_samples[:, :, sample].reshape(-1, encoded_samples.shape[-1]),
                sample_state.reshape(-1, sample_state.shape[-1]),
            ).reshape_as(sample_state)
            sample_state = torch.where(
                imu_mask[:, :, sample, None], candidate, sample_state
            )
        count = valid.sum(dim=-1).clamp_min(1.0)
        interval_duration = (imu_dt * valid).sum(dim=-1)
        mean_acc_raw = (acc.norm(dim=-1) * valid).sum(dim=-1) / count
        mean_gyro_raw = (gyro.norm(dim=-1) * valid).sum(dim=-1) / count
        sensor_context = torch.stack(
            (
                torch.log1p(count),
                torch.log1p(count / interval_duration.clamp_min(1e-4)),
                interval_duration,
                (imu_dt * valid).amax(dim=-1),
                mean_acc_raw / 9.80665,
                torch.log1p(mean_gyro_raw),
            ),
            dim=-1,
        )
        interval_features = self.interval_projection(torch.cat((pooled, sample_state), dim=-1))
        interval_features = interval_features + self.sensor_projection(sensor_context)
        state, hidden = self.temporal_gru(interval_features, initial_hidden)

        bias_scale = state.new_tensor((0.5, 0.5, 0.5, 0.05, 0.05, 0.05))
        bias = torch.tanh(self.bias_head(state)) * bias_scale
        corrected_acc = acc - bias[..., :3].unsqueeze(-2)
        corrected_gyro = gyro - bias[..., 3:].unsqueeze(-2)

        if rotation_camera_from_imu is None:
            rotation_camera_from_imu = torch.eye(
                3, device=imu.device, dtype=imu.dtype
            )
        rotation_camera_from_imu = rotation_camera_from_imu.to(imu)
        corrected_acc = self._rotate(corrected_acc, rotation_camera_from_imu)
        corrected_gyro = self._rotate(corrected_gyro, rotation_camera_from_imu)

        if gravity_camera is None:
            gravity_camera = torch.tensor(
                (0.0, 0.0, -9.80665), device=imu.device, dtype=imu.dtype
            )
        preintegrated = self.preintegrator(
            corrected_acc.float(),
            corrected_gyro.float(),
            imu_dt.float(),
            imu_mask,
            gravity_camera.to(imu).float(),
        )

        acc_norm = corrected_acc.norm(dim=-1)
        gyro_norm = corrected_gyro.norm(dim=-1)
        mean_acc = (acc_norm * valid).sum(dim=-1) / count
        mean_gyro = (gyro_norm * valid).sum(dim=-1) / count
        acc_variance = (
            (acc_norm - mean_acc.unsqueeze(-1)).square() * valid
        ).sum(dim=-1) / count
        gyro_variance = (
            (gyro_norm - mean_gyro.unsqueeze(-1)).square() * valid
        ).sum(dim=-1) / count
        stats = torch.stack(
            (
                mean_acc,
                mean_gyro,
                torch.sqrt(acc_variance + 1e-8),
                torch.sqrt(gyro_variance + 1e-8),
                (imu_dt * valid).sum(dim=-1),
                (imu_dt * valid).amax(dim=-1),
                bias[..., :3].norm(dim=-1),
                bias[..., 3:].norm(dim=-1),
                preintegrated["delta_twist"][..., :3].norm(dim=-1),
                preintegrated["delta_twist"][..., 3:].norm(dim=-1),
            ),
            dim=-1,
        )
        return {
            # Keep the adapter contract independent of the chosen encoder.
            "airimu_feature": self.context_head(state),
            "airimu_correction_gate": torch.zeros(
                state.shape[:-1] + (2,), device=state.device, dtype=state.dtype
            ),
            "bias_state": state,
            "imu_hidden": hidden,
            "estimated_bias": bias,
            "sensor_context": sensor_context,
            "raw_feature": self.raw_head(state),
            "motion_stats": stats,
            "corrected_acc": corrected_acc,
            "corrected_gyro": corrected_gyro,
            **preintegrated,
        }
