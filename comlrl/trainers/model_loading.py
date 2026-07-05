from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM


def load_causal_lm_on_device(
    source: str,
    device: torch.device,
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    """Load a causal LM directly onto a target device when Transformers supports it."""

    target_device = torch.device(device)
    base_kwargs = dict(model_kwargs or {})
    if target_device.type != "cpu":
        direct_kwargs = dict(base_kwargs)
        direct_kwargs.setdefault("low_cpu_mem_usage", True)
        direct_kwargs.setdefault("device_map", {"": str(target_device)})
        try:
            return AutoModelForCausalLM.from_pretrained(source, **direct_kwargs)
        except (ImportError, TypeError, ValueError) as exc:
            if not _is_direct_device_load_option_error(exc):
                raise

    return AutoModelForCausalLM.from_pretrained(source, **base_kwargs).to(target_device)


def _is_direct_device_load_option_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "device_map" in message
        or "low_cpu_mem_usage" in message
        or "accelerate" in message
    )
