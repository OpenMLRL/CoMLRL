from __future__ import annotations

import os
from typing import Optional

import torch

from comlrl.utils.distributed import DistributedContext, init_distributed, local_context


class TorchrunScheduler:
    """Resolve and initialize process-level parallel execution mode."""

    _VALID_MODES = {"auto", "ddp", "scheduler"}

    @staticmethod
    def world_size_from_env() -> int:
        return int(os.environ.get("WORLD_SIZE", "1"))

    @classmethod
    def resolve_mode(cls, requested_mode: Optional[str]) -> str:
        mode = str(requested_mode or "auto").strip().lower()
        if mode not in cls._VALID_MODES:
            raise ValueError("parallel_training must be one of: auto, ddp, scheduler.")

        world_size = cls.world_size_from_env()
        if mode == "auto":
            return "ddp" if world_size > 1 else "scheduler"
        if mode == "scheduler" and world_size > 1:
            raise ValueError(
                "parallel_training='scheduler' requires WORLD_SIZE=1 (single process)."
            )
        return mode

    @staticmethod
    def ddp_context() -> DistributedContext:
        return init_distributed()

    @staticmethod
    def scheduler_context(
        device: Optional[torch.device] = None,
    ) -> DistributedContext:
        return local_context(device)
