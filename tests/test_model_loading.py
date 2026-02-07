import gc
import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from comlrl.trainers.actor_critic import IACTrainer, MAACTrainer
from comlrl.trainers.actor_critic.iac import IACConfig
from comlrl.trainers.actor_critic.maac import MAACConfig
from comlrl.trainers.reinforce import MAGRPOTrainer
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig

MODEL_NAME_05 = os.getenv("COMLRL_TEST_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_NAME_06 = os.getenv("COMLRL_TEST_MODEL_NAME_ALT", "Qwen/Qwen3-0.6B-Instruct")


def _reward_func(*_args, **_kwargs):
    return [0.0]


@pytest.fixture(scope="session")
def tokenizer_05():
    return AutoTokenizer.from_pretrained(MODEL_NAME_05)


@pytest.fixture(scope="session")
def model_05():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME_05)


@pytest.fixture(scope="session")
def model_06():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME_06)


def _cleanup(*objs):
    for obj in objs:
        del obj
    gc.collect()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MAGRPO requires CUDA")
def test_magrpo_loads_homo_from_model_name():
    args = MAGRPOConfig(num_agents=2, num_turns=1, num_generations=2)
    trainer = MAGRPOTrainer(
        model=MODEL_NAME_05,
        num_agents=2,
        reward_func=_reward_func,
        args=args,
    )
    assert trainer.num_agents == 2
    assert trainer.model_name == MODEL_NAME_05
    _cleanup(trainer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MAGRPO requires CUDA")
def test_magrpo_loads_hetero_from_pretrained_list(tokenizer_05, model_05, model_06):
    args = MAGRPOConfig(num_agents=2, num_turns=1, num_generations=2)
    trainer = MAGRPOTrainer(
        agents=[model_05, model_06],
        tokenizer=tokenizer_05,
        reward_func=_reward_func,
        args=args,
    )
    assert trainer.num_agents == 2
    assert trainer.model_name is not None
    _cleanup(trainer)


def test_maac_loads_homo_from_model_name_with_distinct_critic():
    args = MAACConfig(num_agents=2, num_turns=1)
    trainer = MAACTrainer(
        model=MODEL_NAME_05,
        critics=[MODEL_NAME_06],
        reward_func=_reward_func,
        args=args,
    )
    assert len(trainer.agent_models) == 2
    assert trainer.agent_model_name == MODEL_NAME_05
    assert trainer.critic_model_name == MODEL_NAME_06
    _cleanup(trainer)


def test_maac_loads_hetero_from_pretrained_list(tokenizer_05, model_05, model_06):
    args = MAACConfig(num_agents=2, num_turns=1)
    trainer = MAACTrainer(
        agents=[model_05, model_06],
        critics=[model_06],
        tokenizer=tokenizer_05,
        reward_func=_reward_func,
        args=args,
    )
    assert len(trainer.agent_models) == 2
    assert trainer.critic_model is not None
    _cleanup(trainer)


def test_maac_rejects_multiple_critics(tokenizer_05, model_05, model_06):
    args = MAACConfig(num_agents=2, num_turns=1)
    with pytest.raises(ValueError, match="critics length"):
        MAACTrainer(
            agents=[model_05, model_06],
            critics=[model_05, model_06],
            tokenizer=tokenizer_05,
            reward_func=_reward_func,
            args=args,
        )


def test_iac_loads_homo_from_model_name_with_critics(model_05):
    args = IACConfig(num_agents=2, num_turns=1, use_separate_critic=True)
    trainer = IACTrainer(
        model=MODEL_NAME_05,
        critics=[model_05, model_05],
        reward_func=_reward_func,
        args=args,
    )
    assert len(trainer.agent_models) == 2
    assert len(trainer.critic_models) == 2
    _cleanup(trainer)


def test_iac_loads_hetero_shared_value_heads(tokenizer_05, model_05, model_06):
    args = IACConfig(num_agents=2, num_turns=1, use_separate_critic=False)
    trainer = IACTrainer(
        agents=[model_05, model_06],
        tokenizer=tokenizer_05,
        reward_func=_reward_func,
        args=args,
    )
    assert len(trainer.agent_models) == 2
    assert all(c is None for c in trainer.critic_models)
    _cleanup(trainer)


def test_iac_rejects_critics_when_shared_value_heads(tokenizer_05, model_05, model_06):
    args = IACConfig(num_agents=2, num_turns=1, use_separate_critic=False)
    with pytest.raises(ValueError, match="use_separate_critic"):
        IACTrainer(
            agents=[model_05, model_06],
            critics=[model_06, model_05],
            tokenizer=tokenizer_05,
            reward_func=_reward_func,
            args=args,
        )


def test_iac_loads_hetero_with_swapped_critics(tokenizer_05, model_05, model_06):
    args = IACConfig(num_agents=2, num_turns=1, use_separate_critic=True)
    trainer = IACTrainer(
        agents=[model_05, model_06],
        critics=[model_06, model_05],
        tokenizer=tokenizer_05,
        reward_func=_reward_func,
        args=args,
    )
    assert len(trainer.agent_models) == 2
    assert len(trainer.critic_models) == 2
    assert trainer.critic_models[0] is not None
    assert trainer.critic_models[1] is not None
    _cleanup(trainer)
