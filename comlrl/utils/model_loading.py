from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, List


def infer_model_name(source: Any) -> Optional[str]:
    if source is None:
        return None
    if isinstance(source, str):
        return source
    base = getattr(source, "model", source)
    config = getattr(base, "config", None)
    if config is not None:
        name = getattr(config, "_name_or_path", None) or getattr(
            config, "model_type", None
        )
        if name:
            return str(name)
    return base.__class__.__name__


def resolve_model_sources(
    *,
    kind: str,
    model: Optional[Any],
    models: Optional[Sequence[Any]],
    expected_count: int,
    expected_label: Optional[str] = None,
) -> Tuple[List[Any], Optional[str]]:
    if model is not None and models is not None:
        is_name_list = (
            isinstance(models, Sequence)
            and not isinstance(models, (str, bytes))
            and all(isinstance(src, str) for src in models)
        )
        if not is_name_list or len(models) != expected_count:
            label = expected_label or f"num_agents ({expected_count})"
            raise ValueError(
                f"Cannot provide both model and {kind} unless {kind} is a list of {label} model names."
            )
    if model is None and models is None:
        raise ValueError(f"Either model or {kind} must be provided.")
    if expected_count < 1:
        raise ValueError("expected_count must be >= 1.")

    if models is not None:
        if isinstance(models, (str, bytes)) or not isinstance(models, Sequence):
            raise ValueError(f"{kind} must be a non-empty sequence.")
        sources = list(models)
        if len(sources) != expected_count:
            label = expected_label or f"num_agents ({expected_count})"
            raise ValueError(f"{kind} length ({len(sources)}) must match {label}.")
    else:
        sources = [model] * expected_count

    if any(src is None for src in sources):
        raise ValueError(f"{kind} entries must be non-null.")

    model_name = infer_model_name(sources[0]) if sources else None
    return sources, model_name
