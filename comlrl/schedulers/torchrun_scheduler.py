from __future__ import annotations

import os
from typing import Optional

import torch

from comlrl.utils.distributed import DistributedContext, init_distributed, local_context


class TorchrunScheduler:
    """Resolve and initialize process-level parallel execution mode."""

    _VALID_MODES = {"auto", "ddp", "mp"}

    @staticmethod
    def world_size_from_env() -> int:
        try:
            return int(os.environ.get("WORLD_SIZE", "1"))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _missing_ddp_env_vars() -> list[str]:
        required = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
        return [k for k in required if str(os.environ.get(k, "")).strip() == ""]

    @classmethod
    def resolve_mode(cls, requested_mode: Optional[str]) -> str:
        mode = str(requested_mode or "auto").strip().lower()
        if mode not in cls._VALID_MODES:
            raise ValueError("parallel_training must be one of: auto, ddp, mp.")

        world_size = cls.world_size_from_env()
        if mode == "auto":
            if world_size <= 1:
                return "mp"
            # In shared cluster environments WORLD_SIZE may be exported globally.
            # Only switch to DDP when torchrun-style variables are complete.
            return "ddp" if not cls._missing_ddp_env_vars() else "mp"
        if mode == "ddp":
            if world_size <= 1:
                raise ValueError(
                    "parallel_training='ddp' requires WORLD_SIZE>1. "
                    "Use torchrun --nproc_per_node=... to launch."
                )
            missing = cls._missing_ddp_env_vars()
            if missing:
                raise ValueError(
                    "parallel_training='ddp' requires torchrun environment variables. "
                    f"Missing: {', '.join(missing)}."
                )
        if mode == "mp" and world_size > 1:
            raise ValueError(
                "parallel_training='mp' requires WORLD_SIZE=1 (single process)."
            )
        return mode

    @staticmethod
    def ddp_context() -> DistributedContext:
        return init_distributed()

    @staticmethod
    def mp_context(
        device: Optional[torch.device] = None,
    ) -> DistributedContext:
        return local_context(device)
