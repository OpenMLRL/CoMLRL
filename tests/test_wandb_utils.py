from comlrl.utils.wandb_utils import wandb_init_kwargs


def test_wandb_init_kwargs_keeps_every_config_section():
    sections = {
        "dataset": {"name": "data", "type": "task"},
        "agent_model": {"name": "model"},
        "external": {"mode": "feedback", "original_prompt": True},
        "custom": {"key": 1},
    }
    run_config = {"num_agents": 2}
    kwargs = wandb_init_kwargs(
        {"project": "project", "config_sections": sections}, run_config
    )
    assert kwargs["config"] is run_config
    assert run_config == {
        "num_agents": 2,
        **sections,
        "dataset_name": "data",
        "dataset_type": "task",
    }
