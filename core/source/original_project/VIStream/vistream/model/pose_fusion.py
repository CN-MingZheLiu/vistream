"""Causal, pose-level visual-inertial fusion.

The token adapter is useful for geometry features, but a camera token does not
separate rotation, translation, and scale. This module keeps the frozen visual
pose as the prior and applies a bounded error-state correction in SE(3).
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from ..geometry.lie import invert_se3, se3_exp, se3_log


class CausalVisualInertialPoseFusion(nn.Module):
    """Fuse visual relative pose with causal IMU context at pose level."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        hidden_dim = int(getattr(config, "pose_hidden_dim", 128))
        input_dim = 6 + 6 + 3 + config.stats_dim + 256 + config.raw_feature_dim + 6
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU()
        )
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.correction_head = nn.Linear(hidden_dim, 6)
        self.scale_head = nn.Linear(hidden_dim, 1)
        self.gate_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(
            self.gate_head.bias,
            float(getattr(config, "pose_gate_init_logit", -5.0)),
        )

    @staticmethod
    def _relative(poses: torch.Tensor) -> torch.Tensor:
        return invert_se3(poses[:, :-1]) @ poses[:, 1:]

    def _features(
        self,
        visual_twist: torch.Tensor,
        imu: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        values = (
            visual_twist.float(),
            imu["delta_twist"].float(),
            imu["delta_velocity"].float(),
            imu["motion_stats"].float(),
            imu["airimu_feature"].float(),
            imu["raw_feature"].float(),
            imu.get(
                "sensor_context",
                visual_twist.new_zeros(visual_twist.shape[:-1] + (6,)),
            ).float(),
        )
        return self.input_norm(torch.cat(values, dim=-1))

    def _recurrent(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(features)
        return self.temporal_gru(projected, hidden)

    def _correction(
        self,
        visual_twist: torch.Tensor,
        state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raw = self.correction_head(state)
        gate = torch.sigmoid(self.gate_head(state))
        scale_log = torch.tanh(self.scale_head(state)) * float(
            getattr(self.config, "pose_scale_log_max", 2.5)
        )
        scale = torch.exp(scale_log)
        translation_add = torch.tanh(raw[..., :3]) * float(
            getattr(self.config, "pose_translation_add_max", 1.0)
        )
        translation = visual_twist[..., :3] * (scale - 1.0) + translation_add
        rotation = torch.tanh(raw[..., 3:]) * float(
            getattr(self.config, "pose_rotation_max", 0.25)
        )
        stage = getattr(self.config, "stage", "joint")
        if stage == "rotation":
            translation = translation * 0.0
        elif stage in {"translation", "geometry"}:
            rotation = rotation * 0.0
        if stage == "geometry":
            translation = translation * 0.0
        correction = gate * torch.cat((translation, rotation), dim=-1)
        return {
            "correction": correction,
            "gate": gate,
            "scale": scale,
            "translation": gate * translation,
            "rotation": gate * rotation,
        }

    @staticmethod
    def _integrate(
        initial: torch.Tensor,
        relative: torch.Tensor,
    ) -> torch.Tensor:
        poses = [initial[:, 0]]
        for frame in range(relative.shape[1]):
            poses.append(poses[-1] @ relative[:, frame])
        return torch.stack(poses, dim=1)

    @staticmethod
    def _empty_diagnostics(
        native_poses: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, frames = native_poses.shape[:2]
        zeros = native_poses.new_zeros(batch, frames, 1)
        return {
            "camera_gate": zeros,
            "safe_gate": zeros,
            "geometry_gate": zeros,
            "camera_residual_rms": zeros,
            "geometry_residual_rms": zeros,
            "rotation_gain": zeros,
            "translation_gain": native_poses.new_zeros(batch, frames, 3),
            "pose_scale": native_poses.new_ones(batch, frames, 1),
            "pose_correction": native_poses.new_zeros(batch, frames, 6),
        }

    def forward(
        self,
        native_poses: torch.Tensor,
        imu: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if native_poses.ndim != 4 or native_poses.shape[-2:] != (4, 4):
            raise ValueError("native_poses must have shape [B,S,4,4]")
        batch, frames = native_poses.shape[:2]
        if frames <= 1:
            return native_poses, self._empty_diagnostics(native_poses)
        relative_native = self._relative(native_poses.float())
        visual_twist = se3_log(relative_native)
        # IMU interval t belongs to the visual transition t-1 -> t. The first
        # interval is a warm-up sample and has no preceding RGB pose.
        imu_step = {key: value[:, 1:] for key, value in imu.items()}
        if imu_step["delta_twist"].shape[1] != frames - 1:
            raise ValueError("native and IMU frame counts are not aligned")
        features = self._features(visual_twist, imu_step)
        state, hidden = self._recurrent(features)
        correction = self._correction(visual_twist, state)
        corrected_relative = se3_exp(correction["correction"]) @ relative_native
        fused = self._integrate(native_poses.float()[:, :1], corrected_relative)

        zeros = native_poses.new_zeros(batch, 1, 1)
        diagnostics = {
            "camera_gate": torch.cat((zeros, correction["gate"]), dim=1),
            "safe_gate": torch.cat((zeros, correction["gate"]), dim=1),
            "geometry_gate": zeros.expand(batch, frames, 1),
            "camera_residual_rms": torch.cat(
                (
                    zeros,
                    correction["correction"].float().square().mean(
                        dim=-1, keepdim=True
                    ).add(1e-8).sqrt(),
                ),
                dim=1,
            ),
            "geometry_residual_rms": zeros.expand(batch, frames, 1),
            "rotation_gain": torch.cat(
                (zeros, correction["rotation"].norm(dim=-1, keepdim=True)), dim=1
            ),
            "translation_gain": torch.cat(
                (native_poses.new_zeros(batch, 1, 3), correction["translation"]), dim=1
            ),
            "pose_scale": torch.cat(
                (native_poses.new_ones(batch, 1, 1), correction["scale"]), dim=1
            ),
            "pose_correction": torch.cat(
                (native_poses.new_zeros(batch, 1, 6), correction["correction"]), dim=1
            ),
            "pose_fusion_state": hidden,
        }
        return fused.to(native_poses), diagnostics

    def step(
        self,
        native_pose: torch.Tensor,
        previous_native_pose: torch.Tensor,
        previous_fused_pose: torch.Tensor,
        imu_history: Mapping[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        relative_native = invert_se3(previous_native_pose.float()) @ native_pose.float()
        visual_twist = se3_log(relative_native)
        imu_step = {key: value[:, -1:] for key, value in imu_history.items()}
        features = self._features(visual_twist, imu_step)
        state, hidden = self._recurrent(features, hidden)
        correction = self._correction(visual_twist, state)
        relative = se3_exp(correction["correction"]) @ relative_native
        fused_pose = previous_fused_pose.float() @ relative
        diagnostics = {
            "camera_gate": correction["gate"],
            "safe_gate": correction["gate"],
            "geometry_gate": correction["gate"].new_zeros(correction["gate"].shape),
            "camera_residual_rms": correction["correction"].float().square().mean(
                dim=-1, keepdim=True
            ).add(1e-8).sqrt(),
            "geometry_residual_rms": correction["gate"].new_zeros(correction["gate"].shape),
            "rotation_gain": correction["rotation"].norm(dim=-1, keepdim=True),
            "translation_gain": correction["translation"],
            "pose_scale": correction["scale"],
            "pose_correction": correction["correction"],
        }
        return fused_pose.to(native_pose), diagnostics, hidden
