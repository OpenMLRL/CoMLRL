"""Turn a launcher's ``wandb_config`` dict into trainer logging settings."""

import os
from typing import Any, Dict


def wandb_init_kwargs(wandb_config: Any, run_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build ``wandb.init`` kwargs; every ``config_sections`` entry is added to ``run_config``."""
    sections = (
        wandb_config.get("config_sections") if isinstance(wandb_config, dict) else None
    )
    if isinstance(sections, dict):
        run_config.update(sections)
        dataset_section = sections.get("dataset")
        if isinstance(dataset_section, dict):
            if dataset_section.get("name"):
                run_config["dataset_name"] = dataset_section.get("name")
            if dataset_section.get("type"):
                run_config["dataset_type"] = dataset_section.get("type")

    init_kwargs: Dict[str, Any] = {
        "project": wandb_config.get("project"),
        "entity": wandb_config.get("entity"),
        "name": wandb_config.get("name") or wandb_config.get("run_name"),
        "config": run_config,
    }
    wandb_dir = wandb_config.get("dir")
    if wandb_dir is not None:
        os.makedirs(wandb_dir, exist_ok=True)
        init_kwargs["dir"] = wandb_dir
    tags = wandb_config.get("tags")
    if isinstance(tags, list):
        init_kwargs["tags"] = tags
    for key in ("id", "resume"):
        if key in wandb_config:
            init_kwargs[key] = wandb_config[key]
    return init_kwargs


def verbose_from_wandb_config(wandb_config: Any) -> bool:
    """Read ``config_sections.output.verbose`` (default True)."""
    if not isinstance(wandb_config, dict):
        return True
    sections = wandb_config.get("config_sections", {})
    output = sections.get("output", {}) if isinstance(sections, dict) else {}
    if isinstance(output, dict) and "verbose" in output:
        return bool(output.get("verbose"))
    return True
