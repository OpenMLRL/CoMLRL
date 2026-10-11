from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class DistributedContext:
    is_main: bool
    device: torch.device


def local_context(device: Optional[torch.device] = None) -> DistributedContext:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(is_main=True, device=device)


def unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)
