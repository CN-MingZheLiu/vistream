"""Adapter-only checkpoint I/O and optional source-file hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .model.system import VIStream


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_adapter_checkpoint(
    path: str | Path,
    model: VIStream,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    model.assert_backbones_intact()
    payload: dict[str, Any] = {
        "format": "vistream-adapter-v1",
        "step": int(step),
        "stage": model.config.adapter.stage,
        "config": model.config.to_dict(),
        "adapter_state_dict": model.trainable_state_dict(),
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_adapter_checkpoint(
    path: str | Path,
    model: VIStream,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "vistream-adapter-v1":
        raise RuntimeError("not a VIStream adapter-only checkpoint")
    model.load_trainable_state_dict(payload["adapter_state_dict"], strict=strict)
    saved_stage = payload.get("stage")
    if saved_stage is None:
        saved_stage = payload.get("config", {}).get("adapter", {}).get("stage")
    # Optimizer slots are only compatible when the trainable parameter set is
    # unchanged.  Cross-stage resumes restore weights but start fresh slots.
    if (
        optimizer is not None
        and "optimizer_state_dict" in payload
        and (saved_stage is None or saved_stage == model.config.adapter.stage)
    ):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
