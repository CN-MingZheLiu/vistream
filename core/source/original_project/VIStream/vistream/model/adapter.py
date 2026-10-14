"""Causal dual-branch SE(3) adapter inserted before frozen LingBot heads."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FusionAdapterConfig:
    token_dim: int = 2048
    motion_dim: int = 256
    raw_feature_dim: int = 192
    low_rank_dim: int = 128
    num_heads: int = 8
    history_frames: int = 8
    stats_dim: int = 10
    dropout: float = 0.0
    stage: str = "rotation"
    translation_gain_max: float = 0.2
    safe_gate_init_logit: float = -4.0
    # Optional pose-level fusion configuration. These fields are ignored by
    # the legacy token adapter and keep old adapter checkpoints compatible.
    pose_hidden_dim: int = 128
    pose_scale_log_max: float = 2.5
    pose_translation_add_max: float = 1.0
    pose_rotation_max: float = 0.25
    pose_gate_init_logit: float = -5.0


class LieFlowField(nn.Module):
    """Project an SE(3) twist to an analytic first-order image motion basis."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim)
        )

    def forward(
        self,
        twist: torch.Tensor,
        height: int,
        width: int,
        inverse_depth: torch.Tensor | None = None,
        rotation_only: bool = False,
    ) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=twist.device, dtype=twist.dtype),
            torch.linspace(-1.0, 1.0, width, device=twist.device, dtype=twist.dtype),
            indexing="ij",
        )
        x = x.reshape(1, 1, -1)
        y = y.reshape(1, 1, -1)
        tx, ty, tz, wx, wy, wz = twist.unbind(dim=-1)
        tx, ty, tz = (value.unsqueeze(-1) for value in (tx, ty, tz))
        wx, wy, wz = (value.unsqueeze(-1) for value in (wx, wy, wz))
        if inverse_depth is None:
            inverse_depth = torch.ones_like(x.expand(twist.shape[0], twist.shape[1], -1))
        translation_u = (-tx + x * tz) * inverse_depth
        translation_v = (-ty + y * tz) * inverse_depth
        rotation_u = x * y * wx - (1.0 + x.square()) * wy + y * wz
        rotation_v = (1.0 + y.square()) * wx - x * y * wy - x * wz
        u = rotation_u if rotation_only else translation_u + rotation_u
        v = rotation_v if rotation_only else translation_v + rotation_v
        translation_norm = twist[..., :3].norm(dim=-1, keepdim=True)
        rotation_norm = twist[..., 3:].norm(dim=-1, keepdim=True)
        features = torch.stack(
            (
                x.expand_as(u),
                y.expand_as(v),
                u,
                v,
                torch.sqrt(u.square() + v.square() + 1e-8),
                u * x + v * y,
                u * y - v * x,
                tz.expand_as(u),
                translation_norm.expand_as(u),
                rotation_norm.expand_as(u),
            ),
            dim=-1,
        )
        return self.encoder(features)


class CausalSE3FusionAdapter(nn.Module):
    """Navigation-first camera-token and geometry patch-token fusion.

    Three motion tokens are formed per RGB frame: frozen AirIMU context, raw
    micro-motion, and an explicit SE(3) Lie token.  Camera queries attend only
    to motion tokens at or before their timestamp.  Patch modulation uses the
    analytic image motion induced by the same twist.
    """

    motion_tokens_per_frame = 3

    def __init__(self, config: FusionAdapterConfig = FusionAdapterConfig()) -> None:
        super().__init__()
        self.config = config
        c = config
        self.air_projection = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, c.motion_dim), nn.SiLU()
        )
        self.raw_projection = nn.Sequential(
            nn.LayerNorm(c.raw_feature_dim),
            nn.Linear(c.raw_feature_dim, c.motion_dim),
            nn.SiLU(),
        )
        self.lie_projection = nn.Sequential(
            nn.LayerNorm(6 + 3 + c.stats_dim),
            nn.Linear(6 + 3 + c.stats_dim, c.motion_dim),
            nn.SiLU(),
            nn.Linear(c.motion_dim, c.motion_dim),
        )
        self.token_type = nn.Parameter(torch.zeros(3, c.motion_dim))
        self.camera_norm = nn.LayerNorm(c.token_dim)
        self.camera_down = nn.Linear(c.token_dim, c.motion_dim)
        self.camera_attention = nn.MultiheadAttention(
            c.motion_dim,
            c.num_heads,
            dropout=c.dropout,
            batch_first=True,
        )
        self.camera_ffn = nn.Sequential(
            nn.LayerNorm(c.motion_dim),
            nn.Linear(c.motion_dim, c.motion_dim * 2),
            nn.SiLU(),
            nn.Linear(c.motion_dim * 2, c.low_rank_dim),
        )
        self.camera_up = nn.Linear(c.low_rank_dim, c.token_dim)

        self.patch_norms = nn.ModuleList(nn.LayerNorm(c.token_dim) for _ in range(4))
        self.patch_down = nn.ModuleList(
            nn.Linear(c.token_dim, c.low_rank_dim) for _ in range(4)
        )
        self.patch_motion = nn.ModuleList(
            nn.Linear(c.motion_dim, c.low_rank_dim) for _ in range(4)
        )
        self.patch_up = nn.ModuleList(
            nn.Linear(c.low_rank_dim, c.token_dim) for _ in range(4)
        )
        self.flow_field = LieFlowField(c.low_rank_dim)
        self.depth_proxy = nn.ModuleList(nn.Linear(c.token_dim, 1) for _ in range(4))
        self.visual_motion = nn.Linear(c.token_dim, c.motion_dim)
        self.visual_probe = nn.Linear(c.token_dim, 6)
        self.se3_gain = nn.Sequential(
            nn.LayerNorm(c.stats_dim + 12),
            nn.Linear(c.stats_dim + 12, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        self.se3_to_low_rank = nn.Linear(6, c.low_rank_dim)
        self.se3_up = nn.Linear(c.low_rank_dim, c.token_dim)
        self.reliability = nn.Sequential(
            nn.LayerNorm(c.stats_dim + 2),
            nn.Linear(c.stats_dim + 2, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        self.safe_gate = nn.Sequential(
            nn.LayerNorm(c.stats_dim + 2),
            nn.Linear(c.stats_dim + 2, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        nn.init.constant_(self.reliability[-1].bias, -1.5)
        nn.init.constant_(self.safe_gate[-1].bias, c.safe_gate_init_logit)
        self.camera_gate_logit = nn.Parameter(torch.tensor(-1.0))
        self.geometry_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.reset_identity()

    def reset_identity(self) -> None:
        """Make initial fused tokens bit-identical to native tokens."""
        nn.init.zeros_(self.camera_up.weight)
        nn.init.zeros_(self.camera_up.bias)
        nn.init.zeros_(self.se3_up.weight)
        nn.init.zeros_(self.se3_up.bias)
        nn.init.zeros_(self.visual_probe.weight)
        nn.init.zeros_(self.visual_probe.bias)
        for layer in self.depth_proxy:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.patch_up:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _motion_tokens(self, imu: dict[str, torch.Tensor]) -> torch.Tensor:
        air = self.air_projection(imu["airimu_feature"])
        raw = self.raw_projection(imu["raw_feature"])
        lie_input = torch.cat(
            (imu["delta_twist"], imu["delta_velocity"], imu["motion_stats"]), dim=-1
        )
        lie = self.lie_projection(lie_input)
        tokens = torch.stack((air, raw, lie), dim=2)
        return tokens + self.token_type.view(1, 1, 3, -1)

    def _causal_mask(self, frames: int, device: torch.device) -> torch.Tensor:
        query_frame = torch.arange(frames, device=device).unsqueeze(1)
        key_frame = torch.arange(frames, device=device).repeat_interleave(
            self.motion_tokens_per_frame
        ).unsqueeze(0)
        too_new = key_frame > query_frame
        too_old = key_frame < query_frame - self.config.history_frames + 1
        return too_new | too_old

    def _gates(
        self,
        camera_tokens: torch.Tensor,
        motion_tokens: torch.Tensor,
        stats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_delta = torch.zeros_like(camera_tokens)
        visual_delta[:, 1:] = camera_tokens[:, 1:] - camera_tokens[:, :-1]
        visual_motion = F.normalize(self.visual_motion(visual_delta), dim=-1, eps=1e-6)
        lie_motion = F.normalize(motion_tokens[:, :, 2], dim=-1, eps=1e-6)
        air_motion = F.normalize(motion_tokens[:, :, 0], dim=-1, eps=1e-6)
        raw_motion = F.normalize(motion_tokens[:, :, 1], dim=-1, eps=1e-6)
        visual_lie = (visual_motion * lie_motion).sum(dim=-1, keepdim=True)
        air_raw = (air_motion * raw_motion).sum(dim=-1, keepdim=True)
        dynamic = torch.sigmoid(self.reliability(torch.cat((stats, visual_lie, air_raw), dim=-1)))
        safe_gate = torch.sigmoid(self.safe_gate(torch.cat((stats, visual_lie, air_raw), dim=-1)))
        camera = dynamic[..., :1] * torch.sigmoid(self.camera_gate_logit) * safe_gate[..., :1]
        geometry = dynamic[..., 1:] * torch.sigmoid(self.geometry_gate_logit) * safe_gate[..., 1:]
        if self.config.stage not in {"geometry", "joint"}:
            geometry = geometry * 0.0
        return camera, geometry, safe_gate, torch.cat((visual_lie, air_raw), dim=-1)

    def _se3_residual(
        self,
        camera_tokens: torch.Tensor,
        imu: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a bounded, rotation-first innovation in the camera-token space."""
        visual_twist = self.visual_probe(camera_tokens)
        imu_twist = imu["delta_twist"].float()
        motion_stats = imu["motion_stats"].float()
        if imu_twist.shape[1] != visual_twist.shape[1]:
            imu_twist = imu_twist[:, -visual_twist.shape[1]:]
            motion_stats = motion_stats[:, -visual_twist.shape[1]:]
        innovation = visual_twist - imu_twist
        gain_input = torch.cat((motion_stats, visual_twist, imu_twist), dim=-1)
        gains = torch.sigmoid(self.se3_gain(gain_input))
        rotation_gain = gains[..., :1]
        translation_gain = gains[..., 1:] * self.config.translation_gain_max
        if self.config.stage == "rotation":
            translation_gain = translation_gain * 0.0
        scaled = torch.cat(
            (innovation[..., :3] * translation_gain, innovation[..., 3:] * rotation_gain),
            dim=-1,
        )
        residual = self.se3_up(F.silu(self.se3_to_low_rank(scaled)))
        return residual, rotation_gain, translation_gain

    @staticmethod
    def _patch_shape(patch_count: int, image_hw: tuple[int, int], patch_size: int) -> tuple[int, int]:
        height = image_hw[0] // patch_size
        width = image_hw[1] // patch_size
        if height * width != patch_count:
            root = int(round(math.sqrt(patch_count)))
            if root * root != patch_count:
                raise ValueError(
                    f"cannot map {patch_count} patch tokens to image grid {image_hw}"
                )
            return root, root
        return height, width

    def forward(
        self,
        aggregated_tokens: Sequence[torch.Tensor],
        patch_start_idx: int,
        imu: dict[str, torch.Tensor],
        image_hw: tuple[int, int],
        patch_size: int,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        if len(aggregated_tokens) != 4:
            raise ValueError(f"LingBot must provide four feature levels, got {len(aggregated_tokens)}")
        camera_tokens = aggregated_tokens[-1][:, :, 0]
        motion_tokens = self._motion_tokens(imu)
        batch, frames, _, channels = motion_tokens.shape
        if camera_tokens.shape[:2] != (batch, frames):
            raise ValueError("visual and IMU frame counts do not match")
        query = self.camera_down(self.camera_norm(camera_tokens))
        flat_motion = motion_tokens.reshape(batch, frames * 3, channels)
        attended, _ = self.camera_attention(
            query,
            flat_motion,
            flat_motion,
            attn_mask=self._causal_mask(frames, query.device),
            need_weights=False,
        )
        camera_gate, geometry_gate, safe_gate, consistency = self._gates(
            camera_tokens, motion_tokens, imu["motion_stats"]
        )
        se3_residual, rotation_gain, translation_gain = self._se3_residual(camera_tokens, imu)
        camera_residual = (
            self.camera_up(self.camera_ffn(attended)) + se3_residual
        ) * camera_gate

        fused: list[torch.Tensor] = []
        geometry_residual_norms: list[torch.Tensor] = []
        motion_context = attended
        for level, tokens in enumerate(aggregated_tokens):
            output = tokens.clone()
            if level == len(aggregated_tokens) - 1:
                output[:, :, 0] = output[:, :, 0] + camera_residual
            patch = tokens[:, :, patch_start_idx:]
            inverse_depth = F.softplus(
                self.depth_proxy[level](self.patch_norms[level](patch)).squeeze(-1)
            ) + 0.05
            patch_height, patch_width = self._patch_shape(
                patch.shape[-2], image_hw, patch_size
            )
            flow = self.flow_field(
                imu["delta_twist"], patch_height, patch_width,
                inverse_depth=inverse_depth,
                rotation_only=self.config.stage == "rotation",
            )
            hidden = self.patch_down[level](self.patch_norms[level](patch))
            hidden = hidden + self.patch_motion[level](motion_context).unsqueeze(-2) + flow
            residual = self.patch_up[level](F.silu(hidden))
            residual = residual * geometry_gate.unsqueeze(-2)
            output[:, :, patch_start_idx:] = patch + residual
            fused.append(output)
            geometry_residual_norms.append(residual.float().square().mean(dim=(-1, -2)))
        diagnostics = {
            "camera_gate": camera_gate,
            "geometry_gate": geometry_gate,
            "safe_gate": safe_gate,
            "camera_residual_rms": camera_residual.float().square().mean(dim=-1, keepdim=True).add(1e-8).sqrt(),
            "geometry_residual_rms": torch.stack(geometry_residual_norms).mean(dim=0).add(1e-8).sqrt().unsqueeze(-1),
            "rotation_gain": rotation_gain,
            "translation_gain": translation_gain,
            "gate_consistency": consistency,
            "motion_tokens": motion_tokens,
            "native_camera_token": camera_tokens[:, -1:].detach(),
        }
        return fused, diagnostics

    def forward_streaming(
        self,
        aggregated_tokens: Sequence[torch.Tensor],
        patch_start_idx: int,
        imu_history: dict[str, torch.Tensor],
        image_hw: tuple[int, int],
        patch_size: int,
        previous_camera_token: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        """Fuse one visual frame against an all-past IMU motion bank."""
        if len(aggregated_tokens) != 4 or aggregated_tokens[-1].shape[1] != 1:
            raise ValueError("streaming adapter expects four single-frame visual levels")
        camera_token = aggregated_tokens[-1][:, :, 0]
        motion_tokens = self._motion_tokens(imu_history)
        batch, history, _, channels = motion_tokens.shape
        flat_motion = motion_tokens.reshape(batch, history * 3, channels)
        query = self.camera_down(self.camera_norm(camera_token))
        attended, _ = self.camera_attention(
            query, flat_motion, flat_motion, need_weights=False
        )

        if previous_camera_token is None:
            visual_delta = torch.zeros_like(camera_token)
        else:
            visual_delta = camera_token - previous_camera_token.to(camera_token)
        visual_motion = F.normalize(self.visual_motion(visual_delta), dim=-1, eps=1e-6)
        current_motion = motion_tokens[:, -1:]
        lie_motion = F.normalize(current_motion[:, :, 2], dim=-1, eps=1e-6)
        air_motion = F.normalize(current_motion[:, :, 0], dim=-1, eps=1e-6)
        raw_motion = F.normalize(current_motion[:, :, 1], dim=-1, eps=1e-6)
        consistency = torch.cat(
            (
                (visual_motion * lie_motion).sum(dim=-1, keepdim=True),
                (air_motion * raw_motion).sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        dynamic = torch.sigmoid(
            self.reliability(
                torch.cat((imu_history["motion_stats"][:, -1:], consistency), dim=-1)
            )
        )
        safe_gate = torch.sigmoid(self.safe_gate(torch.cat((imu_history["motion_stats"][:, -1:], consistency), dim=-1)))
        camera_gate = dynamic[..., :1] * torch.sigmoid(self.camera_gate_logit) * safe_gate[..., :1]
        geometry_gate = dynamic[..., 1:] * torch.sigmoid(self.geometry_gate_logit) * safe_gate[..., 1:]
        if self.config.stage not in {"geometry", "joint"}:
            geometry_gate = geometry_gate * 0.0
        se3_residual, rotation_gain, translation_gain = self._se3_residual(camera_token, imu_history)
        camera_residual = (self.camera_up(self.camera_ffn(attended)) + se3_residual) * camera_gate

        fused: list[torch.Tensor] = []
        geometry_norms: list[torch.Tensor] = []
        current_twist = imu_history["delta_twist"][:, -1:]
        for level, tokens in enumerate(aggregated_tokens):
            output = tokens.clone()
            if level == 3:
                output[:, :, 0] = output[:, :, 0] + camera_residual
            patch = tokens[:, :, patch_start_idx:]
            inverse_depth = F.softplus(
                self.depth_proxy[level](self.patch_norms[level](patch)).squeeze(-1)
            ) + 0.05
            patch_height, patch_width = self._patch_shape(
                patch.shape[-2], image_hw, patch_size
            )
            flow = self.flow_field(
                current_twist, patch_height, patch_width,
                inverse_depth=inverse_depth,
                rotation_only=self.config.stage == "rotation",
            )
            hidden = self.patch_down[level](self.patch_norms[level](patch))
            hidden = hidden + self.patch_motion[level](attended).unsqueeze(-2)
            hidden = hidden + flow
            residual = self.patch_up[level](F.silu(hidden)) * geometry_gate.unsqueeze(-2)
            output[:, :, patch_start_idx:] = patch + residual
            fused.append(output)
            geometry_norms.append(residual.float().square().mean(dim=(-1, -2)))
        return fused, {
            "camera_gate": camera_gate,
            "geometry_gate": geometry_gate,
            "safe_gate": safe_gate,
            "camera_residual_rms": camera_residual.float().square().mean(dim=-1, keepdim=True).add(1e-8).sqrt(),
            "geometry_residual_rms": torch.stack(geometry_norms).mean(dim=0).add(1e-8).sqrt().unsqueeze(-1),
            "rotation_gain": rotation_gain,
            "translation_gain": translation_gain,
            "gate_consistency": consistency,
            "motion_tokens": current_motion,
            "native_camera_token": camera_token.detach(),
        }
