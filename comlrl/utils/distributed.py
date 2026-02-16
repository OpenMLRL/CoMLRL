from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    is_main: bool
    device: torch.device


def local_context(device: Optional[torch.device] = None) -> DistributedContext:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(
        enabled=False,
        is_main=True,
        device=device,
    )


def unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def is_main_process(ctx: Optional[DistributedContext]) -> bool:
    if ctx is None:
        return True
    return bool(ctx.is_main)


def barrier(ctx: Optional[DistributedContext]) -> None:  # noqa: ARG001
    return


def all_gather_objects(
    obj: Any, ctx: Optional[DistributedContext]
) -> List[Any]:  # noqa: ARG001
    return [obj]


def reduce_metrics_dict(
    metrics: Dict[str, float],
    ctx: Optional[DistributedContext],  # noqa: ARG001
) -> Dict[str, float]:
    return dict(metrics)
