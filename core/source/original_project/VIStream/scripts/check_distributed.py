#!/usr/bin/env python3
"""Minimal CUDA/NCCL/DDP health check with one process per GPU."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def main() -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world < 2:
        raise RuntimeError("launch with torch.distributed.run and at least two processes")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA devices are visible"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    reduced = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(reduced)
    expected = world * (world + 1) / 2
    if float(reduced) != expected:
        raise RuntimeError(f"NCCL all-reduce returned {float(reduced)}, expected {expected}")

    probe = nn.Linear(1, 1, bias=False, device=device)
    with torch.no_grad():
        probe.weight.fill_(1.0)
    ddp = DistributedDataParallel(probe, device_ids=[local_rank])
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
    value = torch.tensor([[float(rank + 1)]], device=device)
    loss = ddp(value).square().mean()
    loss.backward()
    optimizer.step()

    weights = [torch.zeros_like(probe.weight) for _ in range(world)]
    dist.all_gather(weights, probe.weight.detach())
    if not all(torch.equal(weights[0], weight) for weight in weights[1:]):
        raise RuntimeError("DDP optimizer weights diverged across ranks")

    device_names: list[str | None] = [None] * world
    dist.all_gather_object(device_names, torch.cuda.get_device_name(local_rank))
    dist.barrier()
    if rank == 0:
        print(
            json.dumps(
                {
                    "distributed_check": "passed",
                    "backend": dist.get_backend(),
                    "world_size": world,
                    "all_reduce": float(reduced),
                    "ddp_weight": float(weights[0]),
                    "devices": device_names,
                },
                indent=2,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
