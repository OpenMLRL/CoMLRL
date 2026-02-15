from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    is_main: bool
    device: torch.device


def local_context(device: Optional[torch.device] = None) -> DistributedContext:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    return DistributedContext(
        enabled=False,
        rank=0,
        world_size=1,
        local_rank=0,
        is_main=True,
        device=device,
    )


def init_distributed(backend: Optional[str] = None) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    enabled = world_size > 1

    if torch.cuda.is_available():
        if enabled:
            device_count = torch.cuda.device_count()
            if device_count < 1:
                raise RuntimeError(
                    "DDP requested but no CUDA devices are visible to this process."
                )
            if local_rank < 0 or local_rank >= device_count:
                raise ValueError(
                    "Invalid distributed GPU mapping: "
                    f"LOCAL_RANK={local_rank}, visible_cuda_devices={device_count}. "
                    "Make sure nproc_per_node does not exceed visible GPUs."
                )
            torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if enabled else "cuda")
    else:
        device = torch.device("cpu")

    if enabled and not dist.is_initialized():
        backend_name = backend or ("nccl" if device.type == "cuda" else "gloo")
        dist.init_process_group(backend=backend_name, rank=rank, world_size=world_size)

    return DistributedContext(
        enabled=enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_main=(rank == 0),
        device=device,
    )


def wrap_ddp(
    model: torch.nn.Module,
    ctx: DistributedContext,
    *,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    if not ctx.enabled:
        return model
    if isinstance(model, DDP):
        return model
    kwargs = {
        "find_unused_parameters": find_unused_parameters,
    }
    if ctx.device.type == "cuda":
        kwargs["device_ids"] = [ctx.local_rank]
        kwargs["output_device"] = ctx.local_rank
    return DDP(model, **kwargs)


def unwrap_model(model: Any) -> Any:
    return model.module if isinstance(model, DDP) else model


def is_main_process(ctx: Optional[DistributedContext]) -> bool:
    if ctx is None:
        return True
    return bool(ctx.is_main)


def barrier(ctx: Optional[DistributedContext]) -> None:
    if ctx is not None and ctx.enabled and dist.is_initialized():
        dist.barrier()


def all_gather_objects(obj: Any, ctx: Optional[DistributedContext]) -> List[Any]:
    if ctx is None or not ctx.enabled:
        return [obj]
    gathered: List[Any] = [None for _ in range(ctx.world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def reduce_metrics_dict(
    metrics: Dict[str, float],
    ctx: Optional[DistributedContext],
) -> Dict[str, float]:
    """Average scalar metrics across distributed ranks.

    This helper must be called by all ranks in the same order.
    """
    if ctx is None or not ctx.enabled:
        return dict(metrics)
    if not metrics:
        return {}

    keys = sorted(metrics.keys())
    gathered_keys = all_gather_objects(keys, ctx)
    same_keyset = all(k == keys for k in gathered_keys)

    if same_keyset:
        values = torch.tensor(
            [float(metrics[k]) for k in keys], device=ctx.device, dtype=torch.float64
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= float(ctx.world_size)
        reduced = {k: float(values[i].item()) for i, k in enumerate(keys)}
        return reduced if ctx.is_main else {}

    union_keys = sorted({k for key_list in gathered_keys for k in key_list})
    value_tensor = torch.tensor(
        [float(metrics.get(k, 0.0)) for k in union_keys],
        device=ctx.device,
        dtype=torch.float64,
    )
    count_tensor = torch.tensor(
        [1.0 if k in metrics else 0.0 for k in union_keys],
        device=ctx.device,
        dtype=torch.float64,
    )
    dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    count_tensor = torch.clamp(count_tensor, min=1.0)
    averaged = value_tensor / count_tensor
    reduced = {k: float(averaged[i].item()) for i, k in enumerate(union_keys)}
    return reduced if ctx.is_main else {}
