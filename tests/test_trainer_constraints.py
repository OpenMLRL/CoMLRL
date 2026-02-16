from types import SimpleNamespace

import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from comlrl.trainers.actor_critic import IACTrainer, MAACTrainer
from comlrl.trainers.actor_critic.iac import IACConfig
from comlrl.trainers.actor_critic.maac import MAACConfig
from comlrl.trainers.reinforce import MAGRPOTrainer
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig


def _reward_func(*_args, **_kwargs):
    return [0.0]


def _external_transition(**_kwargs):
    return [""] * 1


def _iac_cfg(**kwargs):
    return IACConfig(agent_devices="cpu", critic_devices="cpu", **kwargs)


def _maac_cfg(**kwargs):
    return MAACConfig(agent_devices="cpu", critic_devices="cpu", **kwargs)


def _magrpo_cfg(**kwargs):
    return MAGRPOConfig(agent_devices="cpu", **kwargs)


@pytest.fixture(scope="session")
def dummy_tokenizer():
    return SimpleNamespace(
        pad_token="<pad>",
        eos_token="</s>",
        pad_token_id=0,
        eos_token_id=1,
    )


@pytest.fixture(scope="session")
def tiny_model_a():
    cfg = GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=1,
        n_head=1,
    )
    return GPT2LMHeadModel(cfg)


@pytest.fixture(scope="session")
def tiny_model_b():
    cfg = GPT2Config(
        vocab_size=48,
        n_positions=32,
        n_ctx=32,
        n_embd=24,
        n_layer=1,
        n_head=1,
    )
    return GPT2LMHeadModel(cfg)


@pytest.mark.parametrize(
    "factory, match",
    [
        (
            lambda: IACTrainer(agent_model="dummy", reward_func=None, args=_iac_cfg()),
            "reward_func",
        ),
        (
            lambda: IACTrainer(reward_func=_reward_func, args=_iac_cfg()),
            "Either agent_model or agents",
        ),
        (
            lambda: IACTrainer(
                agents=[object()],
                tokenizer=SimpleNamespace(pad_token="x", eos_token="x", pad_token_id=0),
                reward_func=_reward_func,
                args=_iac_cfg(num_agents=1, num_turns=2),
            ),
            "external_transition",
        ),
        (
            lambda: IACTrainer(
                agent_model="dummy",
                critics=[object()],
                reward_func=_reward_func,
                args=_iac_cfg(use_separate_critic=False),
            ),
            "use_separate_critic",
        ),
        (
            lambda: MAACTrainer(
                agent_model="dummy", reward_func=None, args=_maac_cfg()
            ),
            "reward_func",
        ),
        (
            lambda: MAACTrainer(reward_func=_reward_func, args=_maac_cfg()),
            "Either agent_model or agents",
        ),
        (
            lambda: MAACTrainer(
                agents=[object()],
                tokenizer=SimpleNamespace(pad_token="x", eos_token="x", pad_token_id=0),
                reward_func=_reward_func,
                args=_maac_cfg(num_agents=1, num_turns=2),
            ),
            "external_transition",
        ),
        (
            lambda: MAGRPOTrainer(
                agent_model="dummy", reward_func=None, args=_magrpo_cfg()
            ),
            "reward_func",
        ),
        (
            lambda: MAGRPOTrainer(reward_func=_reward_func, args=_magrpo_cfg()),
            "Either agent_model or agents",
        ),
    ],
    ids=[
        "iac_reward_func",
        "iac_model_or_agents",
        "iac_multiturn_transition",
        "iac_shared_heads_rejects_critics",
        "maac_reward_func",
        "maac_model_or_agents",
        "maac_multiturn_transition",
        "magrpo_reward_func",
        "magrpo_model_or_agents",
    ],
)
def test_trainer_early_constraints(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_iac_separate_critic_requires_critics(dummy_tokenizer, tiny_model_a):
    args = _iac_cfg(num_agents=1, use_separate_critic=True, num_turns=1)
    with pytest.raises(ValueError, match="critics must be provided"):
        IACTrainer(
            agents=[tiny_model_a],
            tokenizer=dummy_tokenizer,
            reward_func=_reward_func,
            args=args,
        )


def test_iac_critic_len_mismatch(dummy_tokenizer, tiny_model_a, tiny_model_b):
    args = _iac_cfg(num_agents=2, use_separate_critic=True, num_turns=1)
    with pytest.raises(ValueError, match="critics length"):
        IACTrainer(
            agents=[tiny_model_a, tiny_model_b],
            critics=[tiny_model_a],
            tokenizer=dummy_tokenizer,
            reward_func=_reward_func,
            args=args,
        )


def test_iac_valid_shared_heads(dummy_tokenizer, tiny_model_a, tiny_model_b):
    args = _iac_cfg(num_agents=2, use_separate_critic=False, num_turns=1)
    trainer = IACTrainer(
        agents=[tiny_model_a, tiny_model_b],
        tokenizer=dummy_tokenizer,
        reward_func=_reward_func,
        args=args,
    )
    assert len(trainer.agents) == 2
    assert len(trainer.critics) == 0


def test_iac_accepts_tokenizer_list(dummy_tokenizer, tiny_model_a, tiny_model_b):
    args = _iac_cfg(num_agents=2, use_separate_critic=False, num_turns=1)
    trainer = IACTrainer(
        agents=[tiny_model_a, tiny_model_b],
        tokenizer=[dummy_tokenizer, dummy_tokenizer],
        reward_func=_reward_func,
        args=args,
    )
    assert len(trainer.tokenizers) == 2


def test_iac_rejects_tokenizer_len_mismatch(
    dummy_tokenizer, tiny_model_a, tiny_model_b
):
    args = _iac_cfg(num_agents=2, use_separate_critic=False, num_turns=1)
    with pytest.raises(ValueError, match="tokenizers length"):
        IACTrainer(
            agents=[tiny_model_a, tiny_model_b],
            tokenizer=[dummy_tokenizer],
            reward_func=_reward_func,
            args=args,
        )


def test_iac_valid_separate_critics(dummy_tokenizer, tiny_model_a, tiny_model_b):
    args = _iac_cfg(num_agents=2, use_separate_critic=True, num_turns=1)
    trainer = IACTrainer(
        agents=[tiny_model_a, tiny_model_b],
        critics=[tiny_model_b, tiny_model_a],
        tokenizer=dummy_tokenizer,
        reward_func=_reward_func,
        args=args,
    )
    assert len(trainer.agents) == 2
    assert len(trainer.critics) == 2
    assert len(trainer.critic_optimizers) == 2


def test_iac_multiturn_with_transition(dummy_tokenizer, tiny_model_a):
    args = _iac_cfg(num_agents=1, use_separate_critic=False, num_turns=2)
    trainer = IACTrainer(
        agents=[tiny_model_a],
        tokenizer=dummy_tokenizer,
        reward_func=_reward_func,
        external_transition=_external_transition,
        args=args,
    )
    assert len(trainer.agents) == 1


def test_iac_multiturn_num_generations_mismatch(dummy_tokenizer, tiny_model_a):
    with pytest.raises(ValueError, match="num_generations"):
        IACTrainer(
            agents=[tiny_model_a],
            tokenizer=dummy_tokenizer,
            reward_func=_reward_func,
            external_transition=_external_transition,
            args=_iac_cfg(num_agents=1, num_turns=2, num_generations=2),
        )


def test_maac_requires_critics(dummy_tokenizer, tiny_model_a):
    args = _maac_cfg(num_agents=1, num_turns=1)
    with pytest.raises(ValueError, match="critics must be provided"):
        MAACTrainer(
            agents=[tiny_model_a],
            tokenizer=dummy_tokenizer,
            reward_func=_reward_func,
            args=args,
        )


def test_maac_critic_len_mismatch(dummy_tokenizer, tiny_model_a, tiny_model_b):
    args = _maac_cfg(num_agents=2, num_turns=1)
    with pytest.raises(ValueError, match="critics length"):
        MAACTrainer(
            agents=[tiny_model_a, tiny_model_b],
            critics=[tiny_model_a, tiny_model_b],
            tokenizer=dummy_tokenizer,
            reward_func=_reward_func,
            args=args,
        )


def test_maac_valid(dummy_tokenizer, tiny_model_a, tiny_model_b):
    args = _maac_cfg(num_agents=2, num_turns=1)
    trainer = MAACTrainer(
        agents=[tiny_model_a, tiny_model_b],
        critics=[tiny_model_a],
        tokenizer=dummy_tokenizer,
        reward_func=_reward_func,
        args=args,
    )
    assert len(trainer.agents) == 2
    assert len(trainer.critics) == 1


def test_maac_multiturn_with_transition(dummy_tokenizer, tiny_model_a):
    args = _maac_cfg(num_agents=1, num_turns=2, num_generations=1)
    trainer = MAACTrainer(
        agents=[tiny_model_a],
        critics=[tiny_model_a],
        tokenizer=dummy_tokenizer,
        reward_func=_reward_func,
        external_transition=_external_transition,
        args=args,
    )
    assert len(trainer.agents) == 1
    assert len(trainer.critics) == 1


def test_maac_multiturn_num_generations_mismatch(dummy_tokenizer, tiny_model_a):
    with pytest.raises(ValueError, match="num_generations"):
        MAACTrainer(
            agents=[tiny_model_a],
            critics=[tiny_model_a],
            tokenizer=dummy_tokenizer,
            reward_func=_reward_func,
            external_transition=_external_transition,
            args=_maac_cfg(num_agents=1, num_turns=2, num_generations=2),
        )


def test_magrpo_requires_transition_for_multiturn(
    dummy_tokenizer, tiny_model_a, tiny_model_b
):
    args = _magrpo_cfg(num_agents=2, num_turns=2, num_generations=2)
    with pytest.raises(ValueError, match="external_transition"):
        MAGRPOTrainer(
            agents=[tiny_model_a, tiny_model_b],
            tokenizer=dummy_tokenizer,
            reward_func=_reward_func,
            args=args,
        )


def test_magrpo_multiturn_with_transition(dummy_tokenizer, tiny_model_a, tiny_model_b):
    args = _magrpo_cfg(num_agents=2, num_turns=2, num_generations=2)
    trainer = MAGRPOTrainer(
        agents=[tiny_model_a, tiny_model_b],
        tokenizer=dummy_tokenizer,
        reward_func=_reward_func,
        external_transition=_external_transition,
        args=args,
    )
    assert trainer.num_agents == 2


def test_magrpo_accepts_tokenizer_list(dummy_tokenizer, tiny_model_a, tiny_model_b):
    args = _magrpo_cfg(num_agents=2, num_turns=1, num_generations=2)
    trainer = MAGRPOTrainer(
        agents=[tiny_model_a, tiny_model_b],
        tokenizer=[dummy_tokenizer, dummy_tokenizer],
        reward_func=_reward_func,
        args=args,
    )
    assert len(trainer.tokenizers) == 2


def test_magrpo_allows_model_and_agent_names(
    dummy_tokenizer, tiny_model_a, monkeypatch
):
    def _fake_from_pretrained(*_args, **_kwargs):
        return tiny_model_a

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", _fake_from_pretrained
    )
    args = _magrpo_cfg(num_agents=2, num_turns=1, num_generations=2)
    trainer = MAGRPOTrainer(
        agent_model="dummy",
        agents=["dummy", "dummy"],
        num_agents=2,
        tokenizer=dummy_tokenizer,
        reward_func=_reward_func,
        args=args,
    )
    assert trainer.num_agents == 2


def test_magrpo_rejects_model_and_agent_conflict(
    dummy_tokenizer, tiny_model_a, monkeypatch
):
    def _fake_from_pretrained(*_args, **_kwargs):
        return tiny_model_a

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", _fake_from_pretrained
    )
    args = _magrpo_cfg(num_agents=2, num_turns=1, num_generations=2)
    with pytest.raises(ValueError, match="conflict"):
        MAGRPOTrainer(
            agent_model="dummy",
            agents=["dummy", "other"],
            num_agents=2,
            tokenizer=dummy_tokenizer,
            reward_func=_reward_func,
            args=args,
        )
