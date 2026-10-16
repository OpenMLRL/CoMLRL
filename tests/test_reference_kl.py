"""Reference-model KL penalties for MAGRPO, IAC and MAAC."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2LMHeadModel

from comlrl.trainers.actor_critic import IACTrainer
from comlrl.trainers.actor_critic.iac import IACConfig
from comlrl.trainers.actor_critic.maac import MAACConfig, MAACTrainer
from comlrl.trainers.reinforce import MAGRPOTrainer
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig
from comlrl.utils.reference_kl import (
    reference_kl_for_sequence,
    resolve_reference_devices,
)

from helpers import tiny_gpt2, zero_reward

CONFIGS = [MAGRPOConfig, IACConfig, MAACConfig]
TRAINERS = [IACTrainer, MAACTrainer, MAGRPOTrainer]


def _actor_critic_trainer(trainer_cls, tokenizer, **options):
    options.update(num_agents=1, num_turns=1, agent_devices="cpu", critic_devices="cpu")
    if trainer_cls is IACTrainer:
        return IACTrainer(
            agents=[tiny_gpt2()],
            tokenizer=tokenizer,
            reward_func=zero_reward,
            args=IACConfig(use_separate_critic=False, **options),
        )
    return MAACTrainer(
        agents=[tiny_gpt2()],
        critics=[tiny_gpt2()],
        tokenizer=tokenizer,
        reward_func=zero_reward,
        args=MAACConfig(**options),
    )


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_reference_kl_config_defaults_off(config_cls):
    config = config_cls()
    assert config.reference_kl_enabled is False
    assert config.reference_devices is None


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_reference_kl_rejects_negative_coef(config_cls):
    with pytest.raises(ValueError, match="reference_kl_coef"):
        config_cls(reference_kl_coef=-0.1)


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_reference_kl_rejects_mismatched_reference_devices(config_cls):
    kwargs = {
        "num_agents": 2,
        "reference_kl_enabled": True,
        "reference_devices": ["cpu", "cpu", "cpu"],
    }
    if config_cls is MAGRPOConfig:
        kwargs["num_generations"] = 2
    with pytest.raises(ValueError, match="reference_devices length"):
        config_cls(**kwargs)


def test_reference_devices_resolve_separately_from_agents():
    args = SimpleNamespace(reference_devices=["cuda:1"])
    fallback = [torch.device("cuda:0")]

    devices = resolve_reference_devices(args, fallback, expected_count=1)

    assert devices == [torch.device("cuda:1")]


def test_magrpo_reference_kl_defaults_to_frozen_actor_copy(stub_tokenizer):
    trainer = MAGRPOTrainer(
        agents=[tiny_gpt2()],
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        args=MAGRPOConfig(
            num_agents=1,
            num_turns=1,
            num_generations=2,
            agent_devices="cpu",
            reference_kl_enabled=True,
        ),
    )

    assert len(trainer.reference_models) == 1
    assert trainer.reference_models[0] is not trainer.agents[0]
    assert all(not p.requires_grad for p in trainer.reference_models[0].parameters())


def test_iac_reference_model_loads_from_actor_source(stub_tokenizer, tmp_path):
    model_dir = tmp_path / "tiny-model"
    tiny_gpt2().save_pretrained(model_dir)

    trainer = IACTrainer(
        agent_model=str(model_dir),
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        args=IACConfig(
            num_agents=1,
            num_turns=1,
            use_separate_critic=False,
            agent_devices="cpu",
            reference_kl_enabled=True,
            reference_devices="cpu",
        ),
    )

    assert len(trainer.reference_models) == 1
    assert isinstance(trainer.reference_models[0], GPT2LMHeadModel)
    assert trainer.reference_models[0] is not trainer.agents[0]
    assert all(not p.requires_grad for p in trainer.reference_models[0].parameters())


def test_reference_kl_for_identical_model_is_zero():
    model = tiny_gpt2()
    sequences = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(sequences)

    kl = reference_kl_for_sequence(
        model,
        model,
        sequences,
        attention_mask,
        prompt_len=2,
        response_len=2,
    )

    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer])
def test_actor_critic_kl_shaping_is_noop_when_disabled(trainer_cls, stub_tokenizer):
    trainer = _actor_critic_trainer(trainer_cls, stub_tokenizer)
    shaped_reward, metadata = trainer._kl_shaped_reward(
        1.0, {"reference_kls": [0.4]}, 0
    )
    assert shaped_reward == pytest.approx(1.0)
    assert metadata == {}


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer])
def test_actor_critic_kl_shaping_penalizes_reward(trainer_cls, stub_tokenizer):
    trainer = _actor_critic_trainer(
        trainer_cls,
        stub_tokenizer,
        reference_kl_enabled=True,
        reference_kl_coef=0.25,
    )

    shaped_reward, metadata = trainer._kl_shaped_reward(
        1.0, {"reference_kls": [0.4]}, 0
    )

    assert shaped_reward == pytest.approx(0.9)
    assert metadata["reference_kl"] == pytest.approx(0.4)
    assert metadata["reference_kl_penalty"] == pytest.approx(0.1)


def test_magrpo_applies_reference_kl_to_returns(stub_tokenizer):
    trainer = MAGRPOTrainer(
        agents=[tiny_gpt2()],
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        args=MAGRPOConfig(
            num_agents=1,
            num_turns=1,
            num_generations=2,
            agent_devices="cpu",
            reference_kl_enabled=True,
            reference_kl_coef=0.5,
        ),
    )

    returns = torch.tensor([1.0, 2.0])
    adjusted = trainer._apply_reference_kl_to_returns(
        returns, {"reference_kls": [0.2, 0.4]}
    )

    assert torch.allclose(adjusted, torch.tensor([0.9, 1.8]))


@pytest.mark.parametrize("trainer_cls", TRAINERS)
def test_reference_kl_values_disabled_or_missing_models(trainer_cls):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(reference_kl_enabled=False)
    assert trainer._reference_kl_values(0, None, None, None, 2, []) == []
    trainer.args.reference_kl_enabled = True
    with pytest.raises(RuntimeError, match="reference models are missing"):
        trainer._reference_kl_values(0, None, None, None, 2, [])


@pytest.mark.parametrize("trainer_cls", TRAINERS)
@pytest.mark.parametrize("response_lens", [[], [2], [2, 1, 3]])
def test_reference_kl_values_keep_order_and_module_hook(
    trainer_cls, response_lens, monkeypatch
):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(reference_kl_enabled=True)
    trainer.reference_models = [object(), object()]
    policy = object()
    ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    calls = []

    def sequence_kl(
        model, reference, sequences, attention_mask, prompt_len, response_len
    ):
        idx = len(calls)
        assert model is policy
        assert reference is trainer.reference_models[1]
        assert torch.equal(sequences, ids[idx : idx + 1])
        assert torch.equal(attention_mask, mask[idx : idx + 1])
        calls.append((prompt_len, response_len))
        return torch.tensor([idx + 0.25, 100.0])

    module = importlib.import_module(trainer_cls.__module__)
    monkeypatch.setattr(module, "reference_kl_for_sequence", sequence_kl)
    values = trainer._reference_kl_values(1, policy, ids, mask, 2, response_lens)
    count = min(len(ids), len(response_lens))
    assert values == [idx + 0.25 for idx in range(count)]
    assert calls == [(2, length) for length in response_lens[:count]]


@pytest.mark.parametrize("trainer_cls", TRAINERS)
def test_reference_kl_values_restore_training_modes_and_rng(trainer_cls):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(reference_kl_enabled=True)
    policy, reference = tiny_gpt2(), tiny_gpt2()
    policy.train()
    reference.eval()
    trainer.reference_models = [reference]
    ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    mask = torch.ones_like(ids)
    rng_before = torch.get_rng_state().clone()

    first = trainer._reference_kl_values(0, policy, ids, mask, 2, [2, 1])
    second = trainer._reference_kl_values(0, policy, ids, mask, 2, [2, 1])

    assert first == second
    assert len(first) == 2 and all(value >= 0.0 for value in first)
    assert policy.training is True
    assert reference.training is False
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(parameter.grad is None for parameter in policy.parameters())
