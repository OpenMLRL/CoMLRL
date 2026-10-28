from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class DistributedContext:
    """Deprecated: CoMLRL trains in one process; kept so existing imports still work."""

    is_main: bool
    device: torch.device


def local_context(device: Optional[torch.device] = None) -> DistributedContext:
    """Deprecated: always the main process, since CoMLRL trains in one process."""
    warnings.warn(
        "local_context and DistributedContext are deprecated and will be removed in "
        "a future release; CoMLRL trains in one process.",
        DeprecationWarning,
        stacklevel=2,
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(is_main=True, device=device)


def unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)
