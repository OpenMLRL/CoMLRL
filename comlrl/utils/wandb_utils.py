"""Turn a launcher's ``wandb_config`` dict into trainer logging settings."""

import os
from typing import Any, Dict


def wandb_init_kwargs(wandb_config: Any, run_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build ``wandb.init`` kwargs; ``config_sections`` are attached to ``run_config``."""
    sections = (
        wandb_config.get("config_sections") if isinstance(wandb_config, dict) else None
    )
    if isinstance(sections, dict):
        dataset_section = sections.get("dataset") or {}
        external_section = sections.get("external") or {}
        run_config.update(
            {
                "dataset": dataset_section,
                "output": sections.get("output") or {},
                "external": external_section,
                "trainer": sections.get("trainer") or {},
            }
        )
        if isinstance(dataset_section, dict):
            if dataset_section.get("name"):
                run_config["dataset_name"] = dataset_section.get("name")
            if dataset_section.get("type"):
                run_config["dataset_type"] = dataset_section.get("type")
        ext_mode = (
            external_section.get("mode") if isinstance(external_section, dict) else None
        )
        if ext_mode:
            run_config["external_mode"] = ext_mode
            for key in ("original_prompt", "previous_response"):
                if key in external_section:
                    run_config[key] = external_section.get(key)

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
