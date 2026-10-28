"""Side-effect-free output path selection shared by training and resume."""

import os


def resolve_output_path(
    directory,
    *,
    cached=None,
    configured=None,
    wandb_config=None,
    default_base=None,
):
    """Keep cached paths verbatim; normalize new paths without expanding ``~``.

    A missing source returns ``None`` unless ``default_base`` is supplied, so
    resume can apply its additional checkpoint-directory fallback.
    """
    if cached:
        return cached
    path = configured
    if not path and isinstance(wandb_config, dict):
        base = wandb_config.get("output_dir")
        if not base:
            sections = wandb_config.get("config_sections") or {}
            output = sections.get("output") if isinstance(sections, dict) else {}
            base = output.get("base_dir") if isinstance(output, dict) else None
            base = base or wandb_config.get("dir")
        if base:
            path = os.path.join(str(base), directory)
    if not path and default_base:
        path = os.path.join(str(default_base), directory)
    return os.path.abspath(str(path)) if path else None
