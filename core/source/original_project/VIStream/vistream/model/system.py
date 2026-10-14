"""End-to-end VIStream wrapper around frozen LingBot and AirIMU backbones."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch
from torch import nn

from ..backbones.airimu import FrozenAirIMUKITTI
from ..geometry.lie import invert_se3
from .adapter import CausalSE3FusionAdapter, FusionAdapterConfig
from .imu import CausalGRUIMUEncoder, DualRateIMUEncoder
from .pose_fusion import CausalVisualInertialPoseFusion


@dataclass
class VIStreamConfig:
    # Paths are resolved relative to the VIStream checkout by _project_path.
    lingbot_source: str = "vendor"
    lingbot_checkpoint: str = "weights/lingbot-map-long.pt"
    airimu_checkpoint: str = "weights/airimu-kitti-best_model.ckpt"
    lingbot_commit: str = "4cd986009b9adeded8a4e740919221940dedeffe"
    image_size: int = 518
    patch_size: int = 14
    warmup_frames: int = 8
    sliding_window_size: int = -1
    kv_cache_sliding_window: int = 64
    kv_cache_cross_frame_special: bool = True
    use_sdpa: bool = True
    enable_depth: bool = True
    enable_points: bool = False
    enable_local_points: bool = False
    # Pose-level causal fusion is the default API. Legacy YAML profiles that
    # omit these fields are mapped to the old token/AirIMU path in
    # ``vistream.config.model_config`` for checkpoint reproducibility.
    fusion_mode: str = "pose"
    imu_architecture: str = "causal_gru"
    imu_state_dim: int = 192
    adapter: FusionAdapterConfig = field(default_factory=FusionAdapterConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrozenBackboneGuard:
    """Detect in-process mutation of frozen tensors using PyTorch versions."""

    def __init__(self, modules: Mapping[str, nn.Module]) -> None:
        self.modules = dict(modules)
        self.versions: dict[str, int] = {}

    def arm(self) -> None:
        self.versions = {
            f"{module_name}.{tensor_name}": tensor._version
            for module_name, module in self.modules.items()
            for tensor_name, tensor in list(module.named_parameters()) + list(module.named_buffers())
        }

    def assert_intact(self) -> None:
        if not self.versions:
            raise RuntimeError("backbone guard has not been armed")
        changed: list[str] = []
        for module_name, module in self.modules.items():
            tensors = list(module.named_parameters()) + list(module.named_buffers())
            for tensor_name, tensor in tensors:
                name = f"{module_name}.{tensor_name}"
                if name not in self.versions or tensor._version != self.versions[name]:
                    changed.append(name)
        if changed:
            raise RuntimeError(
                "frozen backbone tensors changed in-place: " + ", ".join(changed[:8])
            )


def _project_path(path: str | Path) -> Path:
    """Resolve packaged relative paths independently of the caller's cwd."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / candidate).resolve()


def _lingbot_revision(source: Path) -> str | None:
    """Read a Git revision or the revision manifest shipped with a vendor copy."""
    git_dir = source / ".git"
    if git_dir.exists():
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    for manifest in (source / "lingbot_revision.txt", source.parent / "lingbot_revision.txt"):
        if manifest.is_file():
            revision = manifest.read_text(encoding="ascii").strip()
            return revision or None
    return None


def _load_lingbot(config: VIStreamConfig) -> nn.Module:
    source = _project_path(config.lingbot_source)
    checkpoint_path = _project_path(config.lingbot_checkpoint)
    if not (source / "lingbot_map" / "models" / "gct_stream.py").is_file():
        raise FileNotFoundError(f"invalid LingBot source directory: {source}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LingBot checkpoint not found: {checkpoint_path}")
    revision = _lingbot_revision(source)
    if config.lingbot_commit and revision is not None and revision != config.lingbot_commit:
        raise RuntimeError(
            f"LingBot source revision is {revision}, expected {config.lingbot_commit}"
        )
    if config.lingbot_commit and revision is None:
        raise RuntimeError(
            "LingBot source has no Git metadata or vendor/lingbot_revision.txt; "
            "cannot verify the frozen source revision"
        )
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("lingbot_map.models.gct_stream")
    imported_path = Path(module.__file__).resolve()
    if source not in imported_path.parents:
        raise RuntimeError(
            f"LingBot was imported from {imported_path}, expected it under {source}"
        )
    model = module.GCTStream(
        img_size=config.image_size,
        patch_size=config.patch_size,
        enable_camera=True,
        enable_depth=config.enable_depth,
        enable_point=config.enable_points,
        enable_local_point=config.enable_local_points,
        num_frame_for_scale=config.warmup_frames,
        sliding_window_size=config.sliding_window_size,
        kv_cache_sliding_window=config.kv_cache_sliding_window,
        kv_cache_scale_frames=config.warmup_frames,
        kv_cache_cross_frame_special=config.kv_cache_cross_frame_special,
        use_sdpa=config.use_sdpa,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError("LingBot checkpoint does not contain a state dictionary")
    missing, unexpected = model.load_state_dict(state, strict=False)
    required_missing = [
        key for key in missing
        if key.startswith(("aggregator.", "camera_head.", "depth_head.", "point_head."))
    ]
    if required_missing:
        raise RuntimeError(
            "LingBot checkpoint is missing enabled backbone tensors: "
            + ", ".join(required_missing[:8])
        )
    allowed_unexpected = ["local_point_head."]
    if not config.enable_depth:
        allowed_unexpected.append("depth_head.")
    if not config.enable_points:
        allowed_unexpected.append("point_head.")
    if not config.enable_local_points:
        allowed_unexpected.append("local_point_head.")
    invalid_unexpected = [
        key for key in unexpected if not key.startswith(tuple(allowed_unexpected))
    ]
    if invalid_unexpected:
        raise RuntimeError(
            "LingBot checkpoint has incompatible tensors: "
            + ", ".join(invalid_unexpected[:8])
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


class VIStream(nn.Module):
    """Freeze both foundation models and train only pre-head fusion adapters."""

    def __init__(
        self,
        lingbot: nn.Module,
        airimu: FrozenAirIMUKITTI,
        config: VIStreamConfig = VIStreamConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.lingbot = lingbot
        self.airimu = airimu
        if config.imu_architecture == "airimu":
            self.imu_encoder = DualRateIMUEncoder(
                airimu=airimu, raw_feature_dim=config.adapter.raw_feature_dim
            )
        elif config.imu_architecture == "causal_gru":
            self.imu_encoder = CausalGRUIMUEncoder(
                raw_feature_dim=config.adapter.raw_feature_dim,
                state_dim=config.imu_state_dim,
            )
        else:
            raise ValueError(
                f"unknown IMU architecture {config.imu_architecture!r}; "
                "expected 'airimu' or 'causal_gru'"
            )
        self.adapter = CausalSE3FusionAdapter(config.adapter)
        self.pose_fusion = (
            CausalVisualInertialPoseFusion(config.adapter)
            if config.fusion_mode == "pose"
            else None
        )
        if config.fusion_mode not in {"token", "pose"}:
            raise ValueError(
                f"unknown fusion mode {config.fusion_mode!r}; expected 'token' or 'pose'"
            )
        self._freeze_backbones()
        self.guard = FrozenBackboneGuard({"lingbot": lingbot, "airimu": airimu})

    @classmethod
    def from_pretrained(cls, config: VIStreamConfig = VIStreamConfig()) -> "VIStream":
        lingbot = _load_lingbot(config)
        airimu = FrozenAirIMUKITTI.from_checkpoint(_project_path(config.airimu_checkpoint))
        return cls(lingbot=lingbot, airimu=airimu, config=config)

    def _freeze_backbones(self) -> None:
        for module in (self.lingbot, self.airimu):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True) -> "VIStream":
        super().train(mode)
        self.lingbot.eval()
        self.airimu.eval()
        return self

    def arm_backbone_guard(self) -> None:
        self.guard.arm()

    def assert_backbones_intact(self) -> None:
        self.guard.assert_intact()

    def assert_trainable_finite(self) -> None:
        bad = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not torch.isfinite(parameter).all()
        ]
        if bad:
            raise FloatingPointError(
                "non-finite trainable parameters: " + ", ".join(bad[:8])
            )

    def trainable_parameters(self) -> list[nn.Parameter]:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        backbone_ids = {
            id(parameter)
            for module in (self.lingbot, self.airimu)
            for parameter in module.parameters()
        }
        if any(id(parameter) in backbone_ids for parameter in parameters):
            raise RuntimeError("a backbone parameter was accidentally made trainable")
        return parameters

    def configure_stage(self, stage: str) -> None:
        """Select the explicit curriculum stage without touching backbones."""
        valid = {"rotation", "translation", "geometry", "joint"}
        if stage not in valid:
            raise ValueError(f"unknown VIStream stage {stage!r}; expected one of {sorted(valid)}")
        for name, parameter in self.named_parameters():
            if name.startswith(("lingbot.", "airimu.")):
                parameter.requires_grad_(False)
                continue
            if self.config.fusion_mode == "pose" and name.startswith("adapter."):
                parameter.requires_grad_(False)
                continue
            if self.config.fusion_mode == "token" and name.startswith("pose_fusion."):
                parameter.requires_grad_(False)
                continue
            parameter.requires_grad_(True)
            if stage in {"rotation", "translation"} and name.startswith(
                ("adapter.patch_", "adapter.depth_proxy", "adapter.flow_field")
            ):
                parameter.requires_grad_(False)
            if stage == "rotation" and name.startswith("adapter.geometry_gate_logit"):
                parameter.requires_grad_(False)
            if stage == "geometry" and not name.startswith(
                (
                    "adapter.patch_",
                    "adapter.depth_proxy",
                    "adapter.flow_field",
                    "adapter.reliability",
                    "adapter.safe_gate",
                    "adapter.geometry_gate_logit",
                )
            ):
                parameter.requires_grad_(False)
            if stage == "geometry" and name.startswith("imu_encoder."):
                parameter.requires_grad_(False)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state_dict(
        self, state: Mapping[str, torch.Tensor], strict: bool = True
    ) -> None:
        named_parameters = dict(self.named_parameters())
        expected = {name for name, parameter in named_parameters.items() if parameter.requires_grad}
        if strict and set(state) != expected:
            missing = sorted(expected - set(state))
            unexpected = sorted(set(state) - expected)
            raise RuntimeError(
                f"adapter checkpoint mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        with torch.no_grad():
            for name, value in state.items():
                parameter = named_parameters.get(name)
                # A curriculum checkpoint can contain parameters that are
                # frozen in the next stage.  They still need to be restored;
                # only foundation-model tensors are excluded.
                if parameter is None or name.startswith(("lingbot.", "airimu.")):
                    if strict:
                        raise RuntimeError(f"unexpected adapter tensor {name}")
                    continue
                if parameter.shape != value.shape:
                    raise RuntimeError(
                        f"adapter tensor {name} has shape {tuple(value.shape)}, "
                        f"expected {tuple(parameter.shape)}"
                    )
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def _native_heads(
        self,
        tokens: list[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        causal_inference: bool,
        num_frame_per_block: int,
        include_geometry: bool = True,
    ) -> dict[str, torch.Tensor]:
        predictions = self.lingbot._predict_camera(
            tokens,
            causal_inference=causal_inference,
            num_frame_for_scale=self.config.warmup_frames,
            sliding_window_size=self.config.sliding_window_size,
            num_frame_per_block=num_frame_per_block,
        )
        if include_geometry and self.config.enable_depth:
            predictions.update(
                self.lingbot._predict_depth(tokens, images, patch_start_idx)
            )
        if include_geometry and self.config.enable_points:
            predictions.update(
                self.lingbot._predict_points(tokens, images, patch_start_idx)
            )
        if include_geometry and self.config.enable_local_points:
            predictions.update(
                self.lingbot._predict_local_points(tokens, images, patch_start_idx)
            )
        return predictions

    def _decode_cameras(
        self, pose_encoding: torch.Tensor, image_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pose_module = importlib.import_module("lingbot_map.utils.pose_enc")
        extrinsics, intrinsics = pose_module.pose_encoding_to_extri_intri(
            pose_encoding, image_size_hw=image_hw, build_intrinsics=True
        )
        batch, frames = extrinsics.shape[:2]
        world_to_camera = torch.zeros(
            batch, frames, 4, 4, device=extrinsics.device, dtype=extrinsics.dtype
        )
        world_to_camera[..., :3, :] = extrinsics
        world_to_camera[..., 3, 3] = 1.0
        return world_to_camera, invert_se3(world_to_camera), intrinsics

    @staticmethod
    def _native_navigation_diagnostics(
        predictions: dict[str, torch.Tensor],
        world_to_camera: torch.Tensor,
        camera_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
        imu_delta_transform: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        native = dict(predictions)
        shape = predictions["pose_enc"].shape[:2] + (1,)
        zeros = predictions["pose_enc"].new_zeros(shape)
        native.update(
            {
                "world_to_camera": world_to_camera,
                "camera_to_world": camera_to_world,
                "intrinsics": intrinsics,
                "imu_delta_transform": imu_delta_transform.detach(),
                # Native LingBot did not consume IMU; exclude it from the
                # optional IMU-consistency loss used as the oracle baseline.
                "camera_gate": torch.zeros_like(zeros),
                "camera_residual_rms": zeros,
                "geometry_residual_rms": zeros,
            }
        )
        return native

    def _forward_pose_fusion(
        self,
        tokens: list[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        imu_outputs: dict[str, torch.Tensor],
        causal_inference: bool,
        num_frame_per_block: int,
        return_native: bool,
        include_geometry: bool,
    ) -> dict[str, torch.Tensor]:
        if self.pose_fusion is None:
            raise RuntimeError("pose fusion mode has no pose fusion module")
        with torch.no_grad():
            native_predictions = self._native_heads(
                tokens,
                images,
                patch_start_idx,
                causal_inference=causal_inference,
                num_frame_per_block=num_frame_per_block,
                include_geometry=include_geometry,
            )
            native_world_to_camera, native_camera_to_world, intrinsics = self._decode_cameras(
                native_predictions["pose_enc"],
                (images.shape[-2], images.shape[-1]),
            )
            if (
                include_geometry
                and "world_points" not in native_predictions
                and "depth" in native_predictions
            ):
                native_predictions["world_points"] = self.lingbot._unproject_depth_to_world(
                    native_predictions["depth"], native_predictions["pose_enc"]
                )

        camera_to_world, diagnostics = self.pose_fusion(
            native_camera_to_world.detach(), imu_outputs
        )
        predictions = dict(native_predictions)
        predictions.update(
            {
                "world_to_camera": invert_se3(camera_to_world),
                "camera_to_world": camera_to_world,
                "native_camera_to_world": native_camera_to_world.detach(),
                "imu_hidden": imu_outputs.get("imu_hidden"),
                "intrinsics": intrinsics,
                "imu_delta_transform": imu_outputs["delta_transform"],
                "imu_delta_twist": imu_outputs["delta_twist"],
                "imu_delta_velocity": imu_outputs["delta_velocity"],
                **diagnostics,
            }
        )
        if return_native:
            predictions["native_predictions"] = self._native_navigation_diagnostics(
                native_predictions,
                native_world_to_camera,
                native_camera_to_world,
                intrinsics,
                imu_outputs["delta_transform"],
            )
        return predictions

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        causal_inference: bool = True,
        num_frame_per_block: int = 1,
        reset_cache: bool = True,
        return_native: bool = False,
        include_geometry: bool = True,
    ) -> dict[str, torch.Tensor]:
        if reset_cache and hasattr(self.lingbot, "clean_kv_cache"):
            self.lingbot.clean_kv_cache()
        images = batch["images"]
        if images.ndim != 5:
            raise ValueError("images must have shape [B,S,3,H,W]")
        imu_outputs = self.imu_encoder(
            batch["imu"],
            batch["imu_dt"],
            batch["imu_mask"],
            batch.get("rotation_camera_from_imu"),
            batch.get("gravity_camera"),
        )
        # The aggregator is frozen and detached. Frozen heads remain under
        # autograd because their input gradients are required by the adapter.
        with torch.no_grad():
            tokens, patch_start_idx = self.lingbot._aggregate_features(
                images,
                num_frame_for_scale=min(self.config.warmup_frames, images.shape[1]),
                sliding_window_size=self.config.sliding_window_size,
                num_frame_per_block=num_frame_per_block,
            )
            tokens = [token.detach() for token in tokens]
        if self.config.fusion_mode == "pose":
            return self._forward_pose_fusion(
                tokens,
                images,
                patch_start_idx,
                imu_outputs,
                causal_inference,
                num_frame_per_block,
                return_native,
                include_geometry,
            )
        native_predictions = None
        if return_native:
            with torch.no_grad():
                native_predictions = self._native_heads(
                    tokens,
                    images,
                    patch_start_idx,
                    # The teacher is evaluated without a KV cache so it does
                    # not consume or alter the causal cache used by the fused
                    # branch below.  It still sees exactly the same frozen
                    # aggregated tokens.
                    causal_inference=False,
                    num_frame_per_block=num_frame_per_block,
                    include_geometry=False,
                )
                native_world_to_camera, native_camera_to_world, native_intrinsics = self._decode_cameras(
                    native_predictions["pose_enc"], (images.shape[-2], images.shape[-1])
                )
                native_predictions.update(
                    {
                        "world_to_camera": native_world_to_camera,
                        "camera_to_world": native_camera_to_world,
                        "intrinsics": native_intrinsics,
                        "imu_delta_transform": imu_outputs["delta_transform"].detach(),
                        "camera_gate": torch.zeros(
                            native_predictions["pose_enc"].shape[:2] + (1,),
                            device=images.device,
                            dtype=images.dtype,
                        ),
                        "camera_residual_rms": torch.zeros(
                            native_predictions["pose_enc"].shape[:2] + (1,),
                            device=images.device,
                            dtype=images.dtype,
                        ),
                        "geometry_residual_rms": torch.zeros(
                            native_predictions["pose_enc"].shape[:2] + (1,),
                            device=images.device,
                            dtype=images.dtype,
                        ),
                    }
                )
        fused, diagnostics = self.adapter(
            tokens,
            patch_start_idx,
            imu_outputs,
            image_hw=(images.shape[-2], images.shape[-1]),
            patch_size=self.config.patch_size,
        )
        predictions = self._native_heads(
            fused,
            images,
            patch_start_idx,
            causal_inference=causal_inference,
            num_frame_per_block=num_frame_per_block,
            include_geometry=include_geometry,
        )
        if include_geometry and "world_points" not in predictions and "depth" in predictions:
            # lingbot-map-long is released without point_head.  Its native
            # depth head plus pose/intrinsics is the checkpoint-compatible
            # world-point path used for CD/F1.
            predictions["world_points"] = self.lingbot._unproject_depth_to_world(
                predictions["depth"], predictions["pose_enc"]
            )
        world_to_camera, camera_to_world, intrinsics = self._decode_cameras(
            predictions["pose_enc"], (images.shape[-2], images.shape[-1])
        )
        predictions.update(
            {
                "world_to_camera": world_to_camera,
                "camera_to_world": camera_to_world,
                "intrinsics": intrinsics,
                "imu_delta_transform": imu_outputs["delta_transform"],
                "imu_delta_twist": imu_outputs["delta_twist"],
                "imu_delta_velocity": imu_outputs["delta_velocity"],
                **diagnostics,
            }
        )
        if native_predictions is not None:
            predictions["native_predictions"] = native_predictions
        return predictions

    def reset_stream(self) -> None:
        """Reset LingBot KV caches and all fusion-side causal history."""
        if hasattr(self.lingbot, "clean_kv_cache"):
            self.lingbot.clean_kv_cache()
        self._stream_images: list[torch.Tensor] = []
        self._stream_imu: list[torch.Tensor] = []
        self._stream_dt: list[torch.Tensor] = []
        self._stream_mask: list[torch.Tensor] = []
        self._stream_previous_camera: torch.Tensor | None = None
        self._stream_previous_native_pose: torch.Tensor | None = None
        self._stream_fused_pose: torch.Tensor | None = None
        self._stream_pose_hidden: torch.Tensor | None = None
        self._stream_imu_hidden: torch.Tensor | None = None
        self._stream_total_frames = 0
        self._stream_rotation_camera_from_imu: torch.Tensor | None = None
        self._stream_gravity_camera: torch.Tensor | None = None

    @staticmethod
    def _pad_stream_imu(
        values: list[torch.Tensor],
        dts: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        maximum = max(value.shape[1] for value in values)
        batch = values[0].shape[0]
        device, dtype = values[0].device, values[0].dtype
        imu = torch.zeros(batch, len(values), maximum, 6, device=device, dtype=dtype)
        imu_dt = torch.zeros(batch, len(values), maximum, device=device, dtype=dts[0].dtype)
        imu_mask = torch.zeros(batch, len(values), maximum, device=device, dtype=torch.bool)
        for frame, (value, delta, mask) in enumerate(zip(values, dts, masks)):
            count = value.shape[1]
            imu[:, frame, :count] = value
            imu_dt[:, frame, :count] = delta
            imu_mask[:, frame, :count] = mask
        return imu, imu_dt, imu_mask

    @staticmethod
    def _last_frame(predictions: dict[str, torch.Tensor], frames: int) -> dict[str, torch.Tensor]:
        return {
            key: value[:, -1:] if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == frames else value
            for key, value in predictions.items()
        }

    @torch.no_grad()
    def stream_step(
        self,
        image: torch.Tensor,
        imu: torch.Tensor,
        imu_dt: torch.Tensor,
        imu_mask: torch.Tensor,
        rotation_camera_from_imu: torch.Tensor | None = None,
        gravity_camera: torch.Tensor | None = None,
        include_geometry: bool = True,
    ) -> dict[str, torch.Tensor] | None:
        """Consume one completed RGB interval and emit at the RGB rate.

        The first ``warmup_frames - 1`` calls return ``None``.  The warm-up
        call builds native LingBot scale/KV state. Every later call processes
        exactly one new image through LingBot and attends to cached past IMU.
        """
        if image.ndim != 4 or imu.ndim != 3 or imu.shape[-1] != 6:
            raise ValueError("stream_step expects image [B,3,H,W] and imu [B,M,6]")
        if not hasattr(self, "_stream_images"):
            self.reset_stream()
        if rotation_camera_from_imu is not None:
            self._stream_rotation_camera_from_imu = rotation_camera_from_imu
        else:
            rotation_camera_from_imu = self._stream_rotation_camera_from_imu
        if gravity_camera is not None:
            self._stream_gravity_camera = gravity_camera
        else:
            gravity_camera = self._stream_gravity_camera
        self._stream_total_frames += 1
        if self._stream_total_frames <= self.config.warmup_frames:
            self._stream_images.append(image)
        self._stream_imu.append(imu)
        self._stream_dt.append(imu_dt)
        self._stream_mask.append(imu_mask)
        frame_count = self._stream_total_frames
        if frame_count < self.config.warmup_frames:
            return None

        if frame_count == self.config.warmup_frames:
            stacked_imu, stacked_dt, stacked_mask = self._pad_stream_imu(
                self._stream_imu, self._stream_dt, self._stream_mask
            )
            batch = {
                "images": torch.stack(self._stream_images, dim=1),
                "imu": stacked_imu,
                "imu_dt": stacked_dt,
                "imu_mask": stacked_mask,
            }
            if rotation_camera_from_imu is not None:
                batch["rotation_camera_from_imu"] = rotation_camera_from_imu
            if gravity_camera is not None:
                batch["gravity_camera"] = gravity_camera
            predictions = self.forward(
                batch,
                causal_inference=True,
                num_frame_per_block=self.config.warmup_frames,
                include_geometry=include_geometry,
            )
            if self.config.fusion_mode == "pose":
                self._stream_previous_native_pose = predictions[
                    "native_camera_to_world"
                ][:, -1:].detach()
                self._stream_fused_pose = predictions["camera_to_world"][:, -1:].detach()
                self._stream_pose_hidden = predictions.pop(
                    "pose_fusion_state", None
                )
                self._stream_imu_hidden = predictions.pop("imu_hidden", None)
            else:
                self._stream_previous_camera = predictions["native_camera_token"]
            # Images are held only until native KV state is initialized.
            self._stream_images = []
            return self._last_frame(predictions, self.config.warmup_frames)

        history = max(1, self.config.adapter.history_frames)
        self._stream_imu = self._stream_imu[-history:]
        self._stream_dt = self._stream_dt[-history:]
        self._stream_mask = self._stream_mask[-history:]
        stacked_imu, stacked_dt, stacked_mask = self._pad_stream_imu(
            self._stream_imu, self._stream_dt, self._stream_mask
        )
        if (
            self.config.fusion_mode == "pose"
            and self.config.imu_architecture == "causal_gru"
            and self._stream_imu_hidden is not None
        ):
            current_imu, current_dt, current_mask = self._pad_stream_imu(
                self._stream_imu[-1:], self._stream_dt[-1:], self._stream_mask[-1:]
            )
            imu_outputs = self.imu_encoder(
                current_imu,
                current_dt,
                current_mask,
                rotation_camera_from_imu,
                gravity_camera,
                initial_hidden=self._stream_imu_hidden,
            )
        else:
            imu_outputs = self.imu_encoder(
                stacked_imu,
                stacked_dt,
                stacked_mask,
                rotation_camera_from_imu,
                gravity_camera,
            )
        images = image.unsqueeze(1)
        tokens, patch_start_idx = self.lingbot._aggregate_features(
            images,
            num_frame_for_scale=self.config.warmup_frames,
            sliding_window_size=self.config.sliding_window_size,
            num_frame_per_block=1,
        )
        tokens = [token.detach() for token in tokens]
        if self.config.fusion_mode == "pose":
            if (
                self.pose_fusion is None
                or self._stream_previous_native_pose is None
                or self._stream_fused_pose is None
            ):
                raise RuntimeError("pose fusion streaming state was not initialized")
            native_predictions = self._native_heads(
                tokens,
                images,
                patch_start_idx,
                causal_inference=True,
                num_frame_per_block=1,
                include_geometry=include_geometry,
            )
            native_world_to_camera, native_camera_to_world, intrinsics = self._decode_cameras(
                native_predictions["pose_enc"],
                (image.shape[-2], image.shape[-1]),
            )
            camera_to_world, diagnostics, hidden = self.pose_fusion.step(
                native_camera_to_world,
                self._stream_previous_native_pose,
                self._stream_fused_pose,
                imu_outputs,
                self._stream_pose_hidden,
            )
            self._stream_previous_native_pose = native_camera_to_world.detach()
            self._stream_fused_pose = camera_to_world.detach()
            self._stream_pose_hidden = hidden.detach()
            self._stream_imu_hidden = imu_outputs.get("imu_hidden", None)
            if (
                include_geometry
                and "world_points" not in native_predictions
                and "depth" in native_predictions
            ):
                native_predictions["world_points"] = self.lingbot._unproject_depth_to_world(
                    native_predictions["depth"], native_predictions["pose_enc"]
                )
            native_predictions.update(
                {
                    "world_to_camera": invert_se3(camera_to_world),
                    "camera_to_world": camera_to_world,
                    "native_camera_to_world": native_camera_to_world.detach(),
                    "intrinsics": intrinsics,
                    "imu_delta_transform": imu_outputs["delta_transform"][:, -1:],
                    "imu_delta_twist": imu_outputs["delta_twist"][:, -1:],
                    "imu_delta_velocity": imu_outputs["delta_velocity"][:, -1:],
                    **diagnostics,
                }
            )
            return native_predictions
        fused, diagnostics = self.adapter.forward_streaming(
            tokens,
            patch_start_idx,
            imu_outputs,
            image_hw=(image.shape[-2], image.shape[-1]),
            patch_size=self.config.patch_size,
            previous_camera_token=self._stream_previous_camera,
        )
        self._stream_previous_camera = diagnostics["native_camera_token"]
        predictions = self._native_heads(
            fused,
            images,
            patch_start_idx,
            causal_inference=True,
            num_frame_per_block=1,
            include_geometry=include_geometry,
        )
        if include_geometry and "world_points" not in predictions and "depth" in predictions:
            predictions["world_points"] = self.lingbot._unproject_depth_to_world(
                predictions["depth"], predictions["pose_enc"]
            )
        world_to_camera, camera_to_world, intrinsics = self._decode_cameras(
            predictions["pose_enc"], (image.shape[-2], image.shape[-1])
        )
        predictions.update(
            {
                "world_to_camera": world_to_camera,
                "camera_to_world": camera_to_world,
                "intrinsics": intrinsics,
                "imu_delta_transform": imu_outputs["delta_transform"][:, -1:],
                "imu_delta_twist": imu_outputs["delta_twist"][:, -1:],
                "imu_delta_velocity": imu_outputs["delta_velocity"][:, -1:],
                **diagnostics,
            }
        )
        return predictions
