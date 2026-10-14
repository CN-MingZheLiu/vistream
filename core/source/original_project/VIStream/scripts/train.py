#!/usr/bin/env python3
"""Train only VIStream adapters with torchrun/DDP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from vistream.checkpoint import save_adapter_checkpoint
from vistream.config import load_yaml, loss_weights, model_config
from vistream.data import (
    KITTIOdometryWindowDataset,
    TartanAirWindowDataset,
    discover_tartanair_sequences,
    split_tartanair_by_environment,
)
from vistream.losses import vistream_losses
from vistream.geometry.lie import so3_exp
from vistream.model.system import VIStream


def _distributed() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local_rank


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _augment_imu(
    batch: dict[str, Any], settings: dict[str, Any] | None
) -> dict[str, Any]:
    """Randomize SI-unit IMU calibration while preserving RGB interval duration."""
    if not settings or not settings.get("enabled", False):
        return batch
    imu = batch["imu"]
    if not torch.is_tensor(imu) or imu.ndim != 4:
        raise ValueError("IMU augmentation expects batched [B,S,M,6] values")
    batch_size, frames = imu.shape[:2]
    device, dtype = imu.device, imu.dtype
    acc_noise = float(settings.get("acc_noise_std", 0.0))
    gyro_noise = float(settings.get("gyro_noise_std", 0.0))
    acc_bias = float(settings.get("acc_bias_std", 0.0))
    gyro_bias = float(settings.get("gyro_bias_std", 0.0))
    scale_std = float(settings.get("scale_std", 0.0))
    acc_walk = float(settings.get("acc_bias_walk_std", 0.0))
    gyro_walk = float(settings.get("gyro_bias_walk_std", 0.0))
    axis_std = float(settings.get("axis_misalignment_std_deg", 0.0))

    if axis_std > 0.0:
        rotation_vector = torch.randn(batch_size, 3, device=device, dtype=dtype)
        rotation_vector *= axis_std * np.pi / 180.0
        axis_rotation = so3_exp(rotation_vector)
        acc, gyro = imu.split(3, dim=-1)
        acc = torch.einsum("bij,bsmj->bsmi", axis_rotation, acc)
        gyro = torch.einsum("bij,bsmj->bsmi", axis_rotation, gyro)
        imu = torch.cat((acc, gyro), dim=-1)
    noise_std = imu.new_tensor((acc_noise,) * 3 + (gyro_noise,) * 3)
    bias_std = imu.new_tensor((acc_bias,) * 3 + (gyro_bias,) * 3)
    walk_std = imu.new_tensor((acc_walk,) * 3 + (gyro_walk,) * 3)

    scale = 1.0 + torch.randn(
        batch_size, 1, 1, 6, device=device, dtype=dtype
    ) * scale_std
    constant_bias = torch.randn(
        batch_size, 1, 1, 6, device=device, dtype=dtype
    ) * bias_std
    frame_dt = batch["camera_dt"].to(imu).clamp_min(0.0)
    bias_walk = torch.randn(
        batch_size, frames, 1, 6, device=device, dtype=dtype
    ) * walk_std * torch.sqrt(frame_dt).unsqueeze(-1).unsqueeze(-1)
    bias_walk = torch.cumsum(bias_walk, dim=1)
    white_noise = torch.randn_like(imu) * noise_std
    augmented = imu * scale + constant_bias + bias_walk + white_noise
    valid = batch["imu_mask"].unsqueeze(-1)
    batch["imu"] = torch.where(valid, augmented, torch.zeros_like(imu))
    return batch


def _mean_across_workers(value: torch.Tensor, world: int) -> float:
    detached = value.detach().float()
    if world > 1:
        dist.all_reduce(detached, op=dist.ReduceOp.SUM)
        detached /= world
    return float(detached)


def _load_split_manifest(
    path: str | Path, discovered_sequences: list[str]
) -> tuple[list[str], list[str], list[tuple[str, int]], list[tuple[str, int]], dict[str, Any]]:
    manifest_path = Path(path)
    with manifest_path.open() as stream:
        payload = json.load(stream)
    if payload.get("format") != "vistream-tartanair-window-split-v1":
        raise RuntimeError(f"unsupported TartanAir split manifest: {manifest_path}")

    def parse(name: str) -> list[tuple[str, int]]:
        entries = payload.get(name)
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(f"split manifest has no {name} windows")
        parsed = [(str(item["sequence"]), int(item["start"])) for item in entries]
        if len(set(parsed)) != len(parsed):
            raise RuntimeError(f"split manifest contains duplicate {name} windows")
        return parsed

    train_samples = parse("train")
    validation_samples = parse("validation")
    if set(train_samples) & set(validation_samples):
        raise RuntimeError("train and validation manifests overlap")
    discovered = set(discovered_sequences)
    train_sequences = sorted({sequence for sequence, _ in train_samples})
    validation_sequences = sorted({sequence for sequence, _ in validation_samples})
    missing = sorted((set(train_sequences) | set(validation_sequences)) - discovered)
    if missing:
        raise RuntimeError(f"split manifest references unavailable sequences: {missing[:5]}")
    train_environments = {sequence.split("/", 1)[0] for sequence in train_sequences}
    validation_environments = {
        sequence.split("/", 1)[0] for sequence in validation_sequences
    }
    if train_environments & validation_environments:
        raise RuntimeError("train and validation environments must be disjoint")
    return (
        train_sequences,
        validation_sequences,
        train_samples,
        validation_samples,
        payload,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights,
    world: int,
    max_batches: int | None = 64,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _to_device(batch, device)
        predictions = model(batch, return_native=True)
        losses = vistream_losses(predictions, batch, weights)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + _mean_across_workers(value, world)
        count += 1
    model.train()
    return {name: value / max(count, 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vistream_tartanair.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--batch-size-per-gpu", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, help="Override the number of epochs for this explicit stage.")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--window-frames", type=int)
    parser.add_argument("--stride-frames", type=int)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable BF16 autocast for LingBot and adapter forward passes.",
    )
    parser.add_argument(
        "--stage",
        choices=("rotation", "translation", "geometry", "joint"),
        help="Explicit curriculum stage; overrides train.stage and model.adapter.stage.",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run this many optimizer steps with accumulation=1, then exit without validation.",
    )
    parser.add_argument(
        "--smoke-frames",
        type=int,
        help="Override the TartanAir window length for a bounded smoke run (minimum 2).",
    )
    args = parser.parse_args()
    if args.smoke_steps < 0:
        parser.error("--smoke-steps must be non-negative")
    if args.smoke_frames is not None and not args.smoke_steps:
        parser.error("--smoke-frames requires --smoke-steps")
    if args.smoke_frames is not None and args.smoke_frames < 2:
        parser.error("--smoke-frames must be at least 2")
    if args.batch_size_per_gpu is not None and args.batch_size_per_gpu < 1:
        parser.error("--batch-size-per-gpu must be positive")
    if args.gradient_accumulation is not None and args.gradient_accumulation < 1:
        parser.error("--gradient-accumulation must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.window_frames is not None and args.window_frames < 2:
        parser.error("--window-frames must be at least 2")
    if args.stride_frames is not None and args.stride_frames < 1:
        parser.error("--stride-frames must be positive")
    config = load_yaml(args.config)
    stage = args.stage or config.get("train", {}).get(
        "stage", config.get("model", {}).get("adapter", {}).get("stage", "rotation")
    )
    config.setdefault("model", {}).setdefault("adapter", {})["stage"] = stage
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.learning_rate is not None:
        config.setdefault("train", {})["learning_rate"] = args.learning_rate
    if args.window_frames is not None:
        config.setdefault("data", {})["window_frames"] = args.window_frames
    if args.stride_frames is not None:
        config.setdefault("data", {})["stride_frames"] = args.stride_frames
    if args.bf16 is not None:
        config.setdefault("train", {})["bf16"] = args.bf16
    rank, world, local_rank = _distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("VIStream training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    seed = int(config["train"]["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_name = config["data"].get("dataset", "tartanair")
    manifest_payload = None
    split_manifest = None
    if dataset_name == "kitti":
        train_sequences = [str(value) for value in config["data"]["train_sequences"]]
        validation_sequences = [
            str(value) for value in config["data"]["validation_sequences"]
        ]
        if set(train_sequences) & set(validation_sequences):
            raise RuntimeError("KITTI train and validation sequences overlap")
        window_frames = args.smoke_frames or config["data"]["window_frames"]
        stride_frames = (
            args.smoke_frames
            if args.smoke_frames is not None
            else config["data"]["stride_frames"]
        )
        dataset_kwargs = {
            "odometry_root": config["evaluation"]["kitti_odometry_root"],
            "raw_root": config["evaluation"]["kitti_raw_root"],
            "image_size": config["model"]["image_size"],
            "window_frames": window_frames,
            "stride_frames": stride_frames,
            "max_imu_samples": config["data"].get("max_imu_samples", 32),
            "prefer_high_rate_imu": config["data"].get("prefer_high_rate_imu", True),
        }
        train_dataset = ConcatDataset(
            [KITTIOdometryWindowDataset(sequence=sequence, **dataset_kwargs)
             for sequence in train_sequences]
        )
        validation_dataset = ConcatDataset(
            [KITTIOdometryWindowDataset(sequence=sequence, **dataset_kwargs)
             for sequence in validation_sequences]
        )
        manifest_payload = {
            "profile": "kitti-cross-sequence",
            "train_sequences": train_sequences,
            "validation_sequences": validation_sequences,
            "imu_protocols": [
                {
                    "sequence": dataset.sequence,
                    "source": dataset.imu_source,
                    "rate_hz": dataset.imu_rate_hz,
                    "max_interval_samples": dataset.max_interval_imu_samples,
                    "capacity": dataset.max_imu_samples,
                    "calibration": dataset.imu_calibration_source,
                }
                for dataset in (*train_dataset.datasets, *validation_dataset.datasets)
            ],
        }
    elif dataset_name == "tartanair":
        sequences = discover_tartanair_sequences(config["data"]["tartanair_root"])
        split_manifest = config["data"].get("split_manifest")
        train_samples = validation_samples = None
        if split_manifest:
            (
                train_sequences,
                validation_sequences,
                train_samples,
                validation_samples,
                manifest_payload,
            ) = _load_split_manifest(split_manifest, sequences)
        else:
            train_sequences, validation_sequences = split_tartanair_by_environment(
                sequences,
                config["data"]["validation_environments"],
                config["data"]["split_seed"],
            )
        dataset_kwargs = {
            "data_root": config["data"]["tartanair_root"],
            "window_frames": args.smoke_frames or config["data"]["window_frames"],
            "stride_frames": (
                args.smoke_frames if args.smoke_frames is not None
                else config["data"]["stride_frames"]
            ),
            "image_size": config["model"]["image_size"],
            "max_imu_samples": config["data"]["max_imu_samples"],
            "acc_source": config["data"].get("acc_source", "acc"),
        }
        train_dataset = TartanAirWindowDataset(
            sequences=train_sequences, samples=train_samples, **dataset_kwargs
        )
        validation_dataset = TartanAirWindowDataset(
            sequences=validation_sequences, samples=validation_samples, **dataset_kwargs
        )
    else:
        raise ValueError(f"unsupported training dataset {dataset_name!r}")
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world > 1 else None
    validation_sampler = DistributedSampler(validation_dataset, shuffle=False) if world > 1 else None
    loader_kwargs = {
        "batch_size": args.batch_size_per_gpu or config["train"]["batch_size_per_gpu"],
        "num_workers": 0 if args.smoke_steps else config["data"]["num_workers"],
        "pin_memory": True,
        "persistent_workers": False if args.smoke_steps else config["data"]["num_workers"] > 0,
    }
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, shuffle=train_sampler is None, drop_last=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset, sampler=validation_sampler, shuffle=False, **loader_kwargs
    )

    base_model = VIStream.from_pretrained(model_config(config)).to(device)
    base_model.configure_stage(stage)
    base_model.assert_trainable_finite()
    optimizer = torch.optim.AdamW(
        base_model.trainable_parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    start_step = 0
    best_selection_score = float("inf")
    if args.resume:
        from vistream.checkpoint import load_adapter_checkpoint
        payload = load_adapter_checkpoint(
            args.resume, base_model, optimizer=optimizer, strict=False
        )
        base_model.assert_trainable_finite()
        start_step = int(payload["step"])
        best_selection_score = float(
            payload.get("extra", {}).get("best_selection_score", float("inf"))
        )
    model: torch.nn.Module = base_model
    if world > 1:
        # Both backbones stay in eval mode and their buffers are identical on
        # every rank. Re-broadcasting them on each forward would also trip the
        # frozen tensor version guard.
        model = DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
        )
    # Arm after DDP's initial parameter/buffer broadcast.
    base_model.arm_backbone_guard()

    output_dir = Path(args.output_dir or config["train"]["output_dir"])
    configured_accumulation = (
        args.gradient_accumulation
        or int(config["train"]["gradient_accumulation"])
    )
    accumulation = 1 if args.smoke_steps else configured_accumulation
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        split = {
            "train": train_sequences,
            "validation": validation_sequences,
            "train_windows": len(train_dataset),
            "validation_windows": len(validation_dataset),
            "manifest": str(split_manifest) if split_manifest else None,
            "profile": manifest_payload.get("profile") if manifest_payload else None,
            "imu_protocols": manifest_payload.get("imu_protocols", [])
            if manifest_payload
            else [],
        }
        (output_dir / "split.json").write_text(json.dumps(split, indent=2))
        print(
            json.dumps(
                {
                    "training_start": True,
                    "world_size": world,
                    "batch_size_per_gpu": loader_kwargs["batch_size"],
                    "gradient_accumulation": accumulation,
                    "effective_global_batch": world * loader_kwargs["batch_size"] * accumulation,
                    "window_frames": dataset_kwargs["window_frames"],
                    "train_windows": len(train_dataset),
                    "validation_windows": len(validation_dataset),
                    "output_dir": str(output_dir),
                    "stage": stage,
                    "trainable_parameters": sum(
                        parameter.numel() for parameter in base_model.trainable_parameters()
                    ),
                    "imu_protocols": manifest_payload.get("imu_protocols", [])
                    if manifest_payload
                    else [],
                }
            ),
            flush=True,
        )
    weights = loss_weights(config)
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    stop_training = False
    for epoch in range(int(config["train"]["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch_index, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            batch = _augment_imu(batch, config["data"].get("imu_augmentation"))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["train"]["bf16"]):
                predictions = model(batch, return_native=True)
                losses = vistream_losses(
                    predictions, batch, weights, include_per_sample=True
                )
                # Oracle supervision exposes per-sample diagnostics, while
                # the optimizer must always receive one scalar objective.
                loss = losses["total"].mean() / accumulation
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch} batch={batch_index}: "
                    + ", ".join(
                        f"{name}={float(value.detach())}"
                        for name, value in losses.items()
                        if torch.is_tensor(value) and value.ndim == 0
                    )
                )
            loss.backward()
            if (batch_index + 1) % accumulation != 0:
                continue
            nonfinite_gradients = [
                name
                for name, parameter in base_model.named_parameters()
                if parameter.requires_grad
                and parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            if nonfinite_gradients:
                raise FloatingPointError(
                    f"non-finite gradients at epoch={epoch} batch={batch_index}: "
                    + ", ".join(nonfinite_gradients[:20])
                )
            torch.nn.utils.clip_grad_norm_(
                base_model.trainable_parameters(),
                config["train"]["grad_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            base_model.assert_backbones_intact()
            base_model.assert_trainable_finite()
            if rank == 0 and step % 20 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            **{
                                k: float(v.detach())
                                for k, v in losses.items()
                                if torch.is_tensor(v) and v.ndim == 0
                            },
                        }
                    ),
                    flush=True,
                )
            if rank == 0 and step % int(config["train"]["save_every_steps"]) == 0:
                save_adapter_checkpoint(output_dir / f"adapter_{step:08d}.pt", base_model, step, optimizer)
            if args.smoke_steps and step - start_step >= args.smoke_steps:
                stop_training = True
                break
        if stop_training:
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "distributed_smoke": "passed",
                            "world_size": world,
                            "optimizer_steps": step - start_step,
                            "window_frames": dataset_kwargs["window_frames"],
                        }
                    ),
                    flush=True,
                )
            break
        if (epoch + 1) % int(config["train"]["validate_every"]) == 0:
            validation_limit = int(config["train"].get("validation_max_batches", 64))
            scores = validate(
                model,
                validation_loader,
                device,
                weights,
                world,
                max_batches=None if validation_limit <= 0 else validation_limit,
            )
            if rank == 0:
                print(json.dumps({"epoch": epoch + 1, "validation": scores}), flush=True)
                degradation = max(0.0, scores.get("navigation_degradation", 0.0))
                selection_score = scores["navigation"] + 2.0 * degradation
                if selection_score < best_selection_score:
                    best_selection_score = selection_score
                    save_adapter_checkpoint(
                        output_dir / "adapter_best.pt",
                        base_model,
                        step,
                        optimizer,
                        {
                            "epoch": epoch + 1,
                            "validation": scores,
                            "best_selection_score": best_selection_score,
                        },
                    )
                save_adapter_checkpoint(
                    output_dir / "adapter_latest.pt", base_model, step, optimizer,
                    {
                        "epoch": epoch + 1,
                        "validation": scores,
                        "best_selection_score": best_selection_score,
                    },
                )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
