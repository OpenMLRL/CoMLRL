"""Centralized IAC and MAAC train one joint actor on all roles' text."""

from types import SimpleNamespace

import pytest
import torch
import wandb
from datasets import Dataset

from comlrl.trainers.actor_critic import IACConfig, IACTrainer, MAACConfig, MAACTrainer

from helpers import BAD, GOOD, JointAdapter, model_for, task_reward

ITEM = {"id": "task", "prompt": "task prompt"}
OPTIONS = dict(
    collaboration_mode="centralized",
    num_agents=2,
    num_turns=1,
    agent_devices="cpu",
    critic_devices="cpu",
    num_train_epochs=1,
    num_generations=2,
    rollout_buffer_size=2,
    max_new_tokens=8,
    eval_interval=0,
    eval_num_samples=1,
    agent_learning_rate=1e-3,
    critic_learning_rate=1e-3,
    early_termination_threshold=None,
)


def _scripted_actor(tokenizer, prompts):
    """A tiny actor whose samples alternate between a GOOD and a BAD joint response."""
    model = model_for(tokenizer)

    def generate(*, input_ids, num_return_sequences=1, **kwargs):
        prompts.append(tokenizer.decode(input_ids[0]))
        rows = [
            torch.cat(
                [
                    input_ids[0],
                    torch.tensor(
                        tokenizer.encode(GOOD if index % 2 == 0 else BAD),
                        device=input_ids.device,
                    ),
                ]
            )
            for index in range(num_return_sequences)
        ]
        return torch.stack(rows)

    model.generate = generate
    return model


def _trainer(kind, tokenizer, prompts=None, metrics_callback=None, **overrides):
    dataset = Dataset.from_list([ITEM])
    common = dict(
        tokenizer=tokenizer,
        reward_func=task_reward,
        formatters=[lambda _: "left", lambda _: "right"],
        train_dataset=dataset,
        eval_dataset=dataset,
        metrics_callback=metrics_callback,
        centralized_adapter=JointAdapter(),
    )
    options = {**OPTIONS, **overrides}
    actor = _scripted_actor(tokenizer, [] if prompts is None else prompts)
    if kind == "iac":
        return IACTrainer(
            agents=[actor],
            args=IACConfig(use_separate_critic=False, **options),
            **common,
        )
    return MAACTrainer(
        agents=[actor],
        critics=[model_for(tokenizer)],
        args=MAACConfig(**options),
        **common,
    )


@pytest.mark.parametrize("config_cls", [IACConfig, MAACConfig])
def test_configs_validate_the_collaboration_mode(config_cls):
    assert config_cls().collaboration_mode == "decentralized"
    with pytest.raises(ValueError, match="collaboration_mode"):
        config_cls(collaboration_mode="joint")
    with pytest.raises(ValueError, match="num_turns=1"):
        config_cls(collaboration_mode="centralized", num_turns=2)


@pytest.mark.parametrize("kind", ["iac", "maac"])
def test_one_joint_actor_trains_on_the_joint_prompt(kind, tokenizer):
    prompts = []
    trainer = _trainer(kind, tokenizer, prompts)
    assert trainer.num_agents == len(trainer.agents) == 1
    assert trainer.args.num_agents == 2
    before = [p.detach().clone() for p in trainer.agents[0].parameters()]
    trainer.train()
    assert prompts and set(prompts) == {"joint left right"}
    assert trainer.env_step > 0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.agents[0].parameters())
    )


@pytest.mark.parametrize("kind", ["iac", "maac"])
def test_metrics_callback_receives_role_outputs(kind, tokenizer):
    seen = []

    def metrics(rollouts):
        seen.extend((sample.agent_idx, sample.completion) for sample in rollouts)
        return {"samples": float(len(rollouts))}

    trainer = _trainer(kind, tokenizer, metrics_callback=metrics)
    result = trainer.evaluate()
    assert {agent for agent, _ in seen} == {0, 1}
    assert {completion for _, completion in seen} == {"good"}
    assert result["eval/samples"] == 2.0


@pytest.mark.parametrize("critic_type", ["v", "q"])
def test_maac_critic_reads_each_role_like_the_decentralized_critic(
    critic_type, tokenizer
):
    trainer = _trainer("maac", tokenizer, critic_type=critic_type)
    joint = trainer._critic_input_for(ITEM, ["joint left right"], [GOOD])
    roles = trainer._build_critic_input(
        ["left", "right"], None if critic_type == "v" else ["good", "good"]
    )
    assert joint == roles
    assert "[Agent 0] left" in joint and "[Agent 1] right" in joint
    assert ("[Agent 1 action]\ngood" in joint) == (critic_type == "q")


@pytest.mark.parametrize("kind", ["iac", "maac"])
def test_centralized_mode_rejects_several_actors(kind, tokenizer):
    config = (
        IACConfig(use_separate_critic=False, **OPTIONS)
        if kind == "iac"
        else MAACConfig(**OPTIONS)
    )
    trainer_cls = IACTrainer if kind == "iac" else MAACTrainer
    extra = {} if kind == "iac" else {"critics": [model_for(tokenizer)]}
    with pytest.raises(ValueError, match="exactly one actor"):
        trainer_cls(
            agents=[model_for(tokenizer), model_for(tokenizer)],
            tokenizer=tokenizer,
            reward_func=task_reward,
            args=config,
            centralized_adapter=JointAdapter(),
            **extra,
        )


@pytest.mark.parametrize("kind", ["iac", "maac"])
def test_wandb_reports_the_centralized_actor(kind, tokenizer, monkeypatch):
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="run-id", project="project", entity="team")

    monkeypatch.setattr(wandb, "run", None)
    monkeypatch.setattr(wandb, "init", init)
    trainer = _trainer(kind, tokenizer)
    trainer.wandb_config = {"project": "project", "entity": "team"}
    trainer._init_wandb()
    config = calls[0]["config"]
    assert config["collaboration_mode"] == "centralized"
    assert config["num_roles"] == 2
    assert config["num_actor_models"] == config["num_agents"] == 1
