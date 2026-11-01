"""Trainer constructors load agents, critics and tokenizers and validate inputs."""

from types import SimpleNamespace

import pytest

from comlrl.trainers.actor_critic import IACTrainer, MAACTrainer
from comlrl.trainers.actor_critic.iac import IACConfig
from comlrl.trainers.actor_critic.maac import MAACConfig
from comlrl.trainers.pbrl import MADPOTrainer, MARLHFTrainer
from comlrl.trainers.pbrl.madpo import MADPOConfig
from comlrl.trainers.pbrl.marlhf import MARLHFConfig
from comlrl.trainers.reinforce import MAGRPOTrainer
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig

from helpers import tiny_gpt2, word_tokenizer, zero_reward


def _external_transition(**_kwargs):
    return [""]


def _iac_cfg(**kwargs):
    return IACConfig(agent_devices="cpu", critic_devices="cpu", **kwargs)


def _maac_cfg(**kwargs):
    return MAACConfig(agent_devices="cpu", critic_devices="cpu", **kwargs)


def _magrpo_cfg(**kwargs):
    return MAGRPOConfig(agent_devices="cpu", **kwargs)


def _madpo_cfg(**kwargs):
    return MADPOConfig(agent_devices="cpu", **kwargs)


def _marlhf_cfg(**kwargs):
    return MARLHFConfig(agent_devices="cpu", **kwargs)


@pytest.fixture(scope="module")
def tiny_model_a():
    return tiny_gpt2(vocab_size=32, n_embd=16)


@pytest.fixture(scope="module")
def tiny_model_b():
    return tiny_gpt2(vocab_size=48, n_embd=24)


@pytest.fixture(scope="module")
def model_names(tmp_path_factory):
    """Two heterogeneous local checkpoints that load by name like Hub models."""
    root = tmp_path_factory.mktemp("models")
    names = []
    for name, vocab_size, n_embd in (("a", 32, 16), ("b", 48, 24)):
        tiny_gpt2(vocab_size, n_embd).save_pretrained(root / name)
        word_tokenizer().save_pretrained(root / name)
        names.append(str(root / name))
    return names


_PAD_ONLY_TOKENIZER = SimpleNamespace(pad_token="x", eos_token="x", pad_token_id=0)


@pytest.mark.parametrize(
    "factory, match",
    [
        pytest.param(
            lambda: IACTrainer(agent_model="dummy", reward_func=None, args=_iac_cfg()),
            "reward_func",
            id="iac_reward_func",
        ),
        pytest.param(
            lambda: IACTrainer(reward_func=zero_reward, args=_iac_cfg()),
            "Either agent_model or agents",
            id="iac_model_or_agents",
        ),
        pytest.param(
            lambda: IACTrainer(
                agents=[object()],
                tokenizer=_PAD_ONLY_TOKENIZER,
                reward_func=zero_reward,
                args=_iac_cfg(num_agents=1, num_turns=2),
            ),
            "external_transition",
            id="iac_multiturn_transition",
        ),
        pytest.param(
            lambda: IACTrainer(
                agent_model="dummy",
                critics=[object()],
                reward_func=zero_reward,
                args=_iac_cfg(use_separate_critic=False),
            ),
            "use_separate_critic",
            id="iac_shared_heads_rejects_critics",
        ),
        pytest.param(
            lambda: MAACTrainer(
                agent_model="dummy", reward_func=None, args=_maac_cfg()
            ),
            "reward_func",
            id="maac_reward_func",
        ),
        pytest.param(
            lambda: MAACTrainer(reward_func=zero_reward, args=_maac_cfg()),
            "Either agent_model or agents",
            id="maac_model_or_agents",
        ),
        pytest.param(
            lambda: MAACTrainer(
                agents=[object()],
                tokenizer=_PAD_ONLY_TOKENIZER,
                reward_func=zero_reward,
                args=_maac_cfg(num_agents=1, num_turns=2),
            ),
            "external_transition",
            id="maac_multiturn_transition",
        ),
        pytest.param(
            lambda: MAGRPOTrainer(
                agent_model="dummy", reward_func=None, args=_magrpo_cfg()
            ),
            "reward_func",
            id="magrpo_reward_func",
        ),
        pytest.param(
            lambda: MAGRPOTrainer(reward_func=zero_reward, args=_magrpo_cfg()),
            "Either agent_model or agents",
            id="magrpo_model_or_agents",
        ),
        pytest.param(
            lambda: MADPOTrainer(
                agent_model="dummy", reward_func=None, args=_madpo_cfg()
            ),
            "reward_func",
            id="madpo_reward_func",
        ),
        pytest.param(
            lambda: MADPOTrainer(reward_func=zero_reward, args=_madpo_cfg()),
            "Either agent_model or agents",
            id="madpo_model_or_agents",
        ),
        pytest.param(
            lambda: MARLHFTrainer(
                agent_model="dummy", reward_func=None, args=_marlhf_cfg()
            ),
            "reward_func",
            id="marlhf_reward_func",
        ),
        pytest.param(
            lambda: MARLHFTrainer(reward_func=zero_reward, args=_marlhf_cfg()),
            "Either agent_model or agents",
            id="marlhf_model_or_agents",
        ),
    ],
)
def test_trainer_rejects_invalid_inputs_before_loading_models(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_iac_loads_agent_model_by_name(stub_tokenizer, tiny_model_b, model_names):
    trainer = IACTrainer(
        agent_model=model_names[0],
        critics=[tiny_model_b, tiny_model_b],
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        args=_iac_cfg(num_agents=2, num_turns=1, use_separate_critic=True),
    )
    assert len(trainer.agents) == 2
    assert len(trainer.critics) == 2


def test_iac_accepts_agent_names_matching_agent_model(stub_tokenizer, model_names):
    trainer = IACTrainer(
        agent_model=model_names[0],
        agents=[model_names[0], model_names[0]],
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        args=_iac_cfg(num_agents=2, num_turns=1, use_separate_critic=False),
    )
    assert len(trainer.agents) == 2


@pytest.mark.parametrize(
    "agents, match",
    [("both", "conflict"), ("one", "agents length")],
    ids=["names_conflict", "length_mismatch"],
)
def test_iac_rejects_agent_names_inconsistent_with_agent_model(
    stub_tokenizer, model_names, agents, match
):
    with pytest.raises(ValueError, match=match):
        IACTrainer(
            agent_model=model_names[0],
            agents=model_names if agents == "both" else model_names[:1],
            tokenizer=stub_tokenizer,
            reward_func=zero_reward,
            args=_iac_cfg(num_agents=2, num_turns=1, use_separate_critic=False),
        )


def test_iac_separate_critic_requires_critics(stub_tokenizer, tiny_model_a):
    with pytest.raises(ValueError, match="critics must be provided"):
        IACTrainer(
            agents=[tiny_model_a],
            tokenizer=stub_tokenizer,
            reward_func=zero_reward,
            args=_iac_cfg(num_agents=1, use_separate_critic=True, num_turns=1),
        )


def test_iac_rejects_critic_count_mismatch(stub_tokenizer, tiny_model_a, tiny_model_b):
    with pytest.raises(ValueError, match="critics length"):
        IACTrainer(
            agents=[tiny_model_a, tiny_model_b],
            critics=[tiny_model_a],
            tokenizer=stub_tokenizer,
            reward_func=zero_reward,
            args=_iac_cfg(num_agents=2, use_separate_critic=True, num_turns=1),
        )


@pytest.mark.parametrize("heterogeneous", [False, True], ids=["homo", "hetero"])
def test_iac_shared_heads_build_no_critics(
    stub_tokenizer, tiny_model_a, tiny_model_b, heterogeneous
):
    trainer = IACTrainer(
        agents=[tiny_model_a, tiny_model_b if heterogeneous else tiny_model_a],
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        args=_iac_cfg(num_agents=2, use_separate_critic=False, num_turns=1),
    )
    assert len(trainer.agents) == 2
    assert len(trainer.critics) == 0


@pytest.mark.parametrize("swapped", [False, True], ids=["match", "swapped"])
def test_iac_separate_critics_get_one_optimizer_each(
    stub_tokenizer, tiny_model_a, tiny_model_b, swapped
):
    critics = [tiny_model_b, tiny_model_a] if swapped else [tiny_model_a, tiny_model_b]
    trainer = IACTrainer(
        agents=[tiny_model_a, tiny_model_b],
        critics=critics,
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        args=_iac_cfg(num_agents=2, use_separate_critic=True, num_turns=1),
    )
    assert len(trainer.agents) == 2
    assert len(trainer.critics) == 2
    assert len(trainer.critic_optimizers) == 2


def test_iac_accepts_tokenizer_list(stub_tokenizer, tiny_model_a, tiny_model_b):
    trainer = IACTrainer(
        agents=[tiny_model_a, tiny_model_b],
        tokenizer=[stub_tokenizer, stub_tokenizer],
        reward_func=zero_reward,
        args=_iac_cfg(num_agents=2, use_separate_critic=False, num_turns=1),
    )
    assert len(trainer.tokenizers) == 2


def test_iac_rejects_tokenizer_count_mismatch(
    stub_tokenizer, tiny_model_a, tiny_model_b
):
    with pytest.raises(ValueError, match="tokenizers length"):
        IACTrainer(
            agents=[tiny_model_a, tiny_model_b],
            tokenizer=[stub_tokenizer],
            reward_func=zero_reward,
            args=_iac_cfg(num_agents=2, use_separate_critic=False, num_turns=1),
        )


def test_iac_multiturn_accepts_external_transition(stub_tokenizer, tiny_model_a):
    trainer = IACTrainer(
        agents=[tiny_model_a],
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        external_transition=_external_transition,
        args=_iac_cfg(num_agents=1, use_separate_critic=False, num_turns=2),
    )
    assert len(trainer.agents) == 1


def test_maac_loads_agent_and_critic_by_name(model_names):
    trainer = MAACTrainer(
        agent_model=model_names[0],
        critics=[model_names[1]],
        reward_func=zero_reward,
        args=_maac_cfg(num_agents=2, num_turns=1),
    )
    assert len(trainer.agents) == 2
    assert len(trainer.critics) == 1


def test_maac_requires_critics(stub_tokenizer, tiny_model_a):
    with pytest.raises(ValueError, match="critics must be provided"):
        MAACTrainer(
            agents=[tiny_model_a],
            tokenizer=stub_tokenizer,
            reward_func=zero_reward,
            args=_maac_cfg(num_agents=1, num_turns=1),
        )


def test_maac_rejects_more_than_one_critic(stub_tokenizer, tiny_model_a, tiny_model_b):
    with pytest.raises(ValueError, match="critics length"):
        MAACTrainer(
            agents=[tiny_model_a, tiny_model_b],
            critics=[tiny_model_a, tiny_model_b],
            tokenizer=stub_tokenizer,
            reward_func=zero_reward,
            args=_maac_cfg(num_agents=2, num_turns=1),
        )


def test_maac_shares_one_critic_across_agents(
    stub_tokenizer, tiny_model_a, tiny_model_b
):
    trainer = MAACTrainer(
        agents=[tiny_model_a, tiny_model_b],
        critics=[tiny_model_a],
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        args=_maac_cfg(num_agents=2, num_turns=1),
    )
    assert len(trainer.agents) == 2
    assert len(trainer.critics) == 1


def test_maac_multiturn_accepts_external_transition(stub_tokenizer, tiny_model_a):
    trainer = MAACTrainer(
        agents=[tiny_model_a],
        critics=[tiny_model_a],
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        external_transition=_external_transition,
        args=_maac_cfg(num_agents=1, num_turns=2, num_generations=1),
    )
    assert len(trainer.agents) == 1
    assert len(trainer.critics) == 1


def test_magrpo_loads_agent_model_by_name(model_names):
    trainer = MAGRPOTrainer(
        agent_model=model_names[0],
        num_agents=2,
        reward_func=zero_reward,
        args=_magrpo_cfg(num_agents=2, num_turns=1, num_generations=2),
    )
    assert trainer.num_agents == 2
    assert trainer.model_name == model_names[0]


def test_magrpo_accepts_heterogeneous_agents(
    stub_tokenizer, tiny_model_a, tiny_model_b
):
    trainer = MAGRPOTrainer(
        agents=[tiny_model_a, tiny_model_b],
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        args=_magrpo_cfg(num_agents=2, num_turns=1, num_generations=2),
    )
    assert trainer.num_agents == 2
    assert trainer.model_name is not None


def test_magrpo_accepts_tokenizer_list(stub_tokenizer, tiny_model_a, tiny_model_b):
    trainer = MAGRPOTrainer(
        agents=[tiny_model_a, tiny_model_b],
        tokenizer=[stub_tokenizer, stub_tokenizer],
        reward_func=zero_reward,
        args=_magrpo_cfg(num_agents=2, num_turns=1, num_generations=2),
    )
    assert len(trainer.tokenizers) == 2


def test_magrpo_multiturn_requires_external_transition(
    stub_tokenizer, tiny_model_a, tiny_model_b
):
    with pytest.raises(ValueError, match="external_transition"):
        MAGRPOTrainer(
            agents=[tiny_model_a, tiny_model_b],
            tokenizer=stub_tokenizer,
            reward_func=zero_reward,
            args=_magrpo_cfg(num_agents=2, num_turns=2, num_generations=2),
        )


def test_magrpo_multiturn_accepts_external_transition(
    stub_tokenizer, tiny_model_a, tiny_model_b
):
    trainer = MAGRPOTrainer(
        agents=[tiny_model_a, tiny_model_b],
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        external_transition=_external_transition,
        args=_magrpo_cfg(num_agents=2, num_turns=2, num_generations=2),
    )
    assert trainer.num_agents == 2


def test_magrpo_accepts_agent_names_matching_agent_model(stub_tokenizer, model_names):
    trainer = MAGRPOTrainer(
        agent_model=model_names[0],
        agents=[model_names[0], model_names[0]],
        num_agents=2,
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        args=_magrpo_cfg(num_agents=2, num_turns=1, num_generations=2),
    )
    assert trainer.num_agents == 2


def test_magrpo_rejects_agent_names_conflicting_with_agent_model(
    stub_tokenizer, model_names
):
    with pytest.raises(ValueError, match="conflict"):
        MAGRPOTrainer(
            agent_model=model_names[0],
            agents=model_names,
            num_agents=2,
            tokenizer=stub_tokenizer,
            reward_func=zero_reward,
            args=_magrpo_cfg(num_agents=2, num_turns=1, num_generations=2),
        )
