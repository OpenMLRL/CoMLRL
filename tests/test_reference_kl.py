import importlib
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from comlrl.trainers.actor_critic import IACTrainer
from comlrl.trainers.actor_critic.iac import IACConfig
from comlrl.trainers.actor_critic.maac import MAACConfig, MAACTrainer
from comlrl.trainers.reinforce import MAGRPOTrainer
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig
from comlrl.utils.reference_kl import (
    reference_kl_for_sequence,
    resolve_reference_devices,
)


def _tiny_model(vocab_size: int = 32) -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=1,
        n_head=1,
    )
    return GPT2LMHeadModel(cfg)


def _dummy_tokenizer():
    return SimpleNamespace(
        pad_token="<pad>",
        eos_token="</s>",
        pad_token_id=0,
        eos_token_id=1,
    )


def _reward_func(*_args, **_kwargs):
    return [0.0]


def test_reference_kl_config_defaults_off():
    magrpo = MAGRPOConfig()
    iac = IACConfig()
    maac = MAACConfig()

    assert magrpo.reference_kl_enabled is False
    assert iac.reference_kl_enabled is False
    assert maac.reference_kl_enabled is False
    assert magrpo.reference_devices is None
    assert iac.reference_devices is None
    assert maac.reference_devices is None


@pytest.mark.parametrize("config_cls", [MAGRPOConfig, IACConfig, MAACConfig])
def test_reference_kl_rejects_negative_coef(config_cls):
    with pytest.raises(ValueError, match="reference_kl_coef"):
        config_cls(reference_kl_coef=-0.1)


@pytest.mark.parametrize("config_cls", [MAGRPOConfig, IACConfig, MAACConfig])
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


def test_reference_devices_can_be_resolved_separately_from_agents():
    args = SimpleNamespace(reference_devices=["cuda:1"])
    fallback = [torch.device("cuda:0")]

    devices = resolve_reference_devices(args, fallback, expected_count=1)

    assert devices == [torch.device("cuda:1")]


def test_reference_kl_enabled_uses_self_reference_by_default():
    cfg = MAGRPOConfig(
        num_agents=1,
        num_turns=1,
        num_generations=2,
        agent_devices="cpu",
        reference_kl_enabled=True,
    )
    trainer = MAGRPOTrainer(
        agents=[_tiny_model()],
        tokenizer=_dummy_tokenizer(),
        reward_func=_reward_func,
        args=cfg,
    )

    assert len(trainer.reference_models) == 1
    assert trainer.reference_models[0] is not trainer.agents[0]
    assert all(not p.requires_grad for p in trainer.reference_models[0].parameters())


def test_reference_model_loads_from_actor_source(tmp_path):
    model_dir = tmp_path / "tiny-model"
    _tiny_model().save_pretrained(model_dir)

    trainer = IACTrainer(
        agent_model=str(model_dir),
        tokenizer=_dummy_tokenizer(),
        reward_func=_reward_func,
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
    model = _tiny_model()
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


def test_reference_kl_disabled_actor_critic_shaping_is_noop():
    iac = IACTrainer(
        agents=[_tiny_model()],
        tokenizer=_dummy_tokenizer(),
        reward_func=_reward_func,
        args=IACConfig(
            num_agents=1,
            num_turns=1,
            use_separate_critic=False,
            agent_devices="cpu",
            critic_devices="cpu",
        ),
    )
    shaped_reward, metadata = iac._kl_shaped_reward(1.0, {"reference_kls": [0.4]}, 0)
    assert shaped_reward == pytest.approx(1.0)
    assert metadata == {}

    maac = MAACTrainer(
        agents=[_tiny_model()],
        critics=[_tiny_model()],
        tokenizer=_dummy_tokenizer(),
        reward_func=_reward_func,
        args=MAACConfig(
            num_agents=1,
            num_turns=1,
            agent_devices="cpu",
            critic_devices="cpu",
        ),
    )
    shaped_reward, metadata = maac._kl_shaped_reward(1.0, {"reference_kls": [0.4]}, 0)
    assert shaped_reward == pytest.approx(1.0)
    assert metadata == {}


def test_magrpo_applies_reference_kl_to_returns():
    args = MAGRPOConfig(
        num_agents=1,
        num_turns=1,
        num_generations=2,
        agent_devices="cpu",
        reference_kl_enabled=True,
        reference_kl_coef=0.5,
    )
    trainer = MAGRPOTrainer(
        agents=[_tiny_model()],
        tokenizer=_dummy_tokenizer(),
        reward_func=_reward_func,
        args=args,
    )

    returns = torch.tensor([1.0, 2.0])
    adjusted = trainer._apply_reference_kl_to_returns(
        returns, {"reference_kls": [0.2, 0.4]}
    )

    assert torch.allclose(adjusted, torch.tensor([0.9, 1.8]))


def test_iac_uses_reference_kl_shaped_reward():
    args = IACConfig(
        num_agents=1,
        num_turns=1,
        use_separate_critic=False,
        agent_devices="cpu",
        critic_devices="cpu",
        reference_kl_enabled=True,
        reference_kl_coef=0.25,
    )
    trainer = IACTrainer(
        agents=[_tiny_model()],
        tokenizer=_dummy_tokenizer(),
        reward_func=_reward_func,
        args=args,
    )

    shaped_reward, metadata = trainer._kl_shaped_reward(
        1.0, {"reference_kls": [0.4]}, 0
    )

    assert shaped_reward == pytest.approx(0.9)
    assert metadata["reference_kl"] == pytest.approx(0.4)
    assert metadata["reference_kl_penalty"] == pytest.approx(0.1)


def test_maac_uses_reference_kl_shaped_reward():
    args = MAACConfig(
        num_agents=1,
        num_turns=1,
        agent_devices="cpu",
        critic_devices="cpu",
        reference_kl_enabled=True,
        reference_kl_coef=0.25,
    )
    trainer = MAACTrainer(
        agents=[_tiny_model()],
        critics=[_tiny_model()],
        tokenizer=_dummy_tokenizer(),
        reward_func=_reward_func,
        args=args,
    )

    shaped_reward, metadata = trainer._kl_shaped_reward(
        1.0, {"reference_kls": [0.4]}, 0
    )

    assert shaped_reward == pytest.approx(0.9)
    assert metadata["reference_kl"] == pytest.approx(0.4)
    assert metadata["reference_kl_penalty"] == pytest.approx(0.1)


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer, MAGRPOTrainer])
def test_sequence_kl_keeps_disabled_and_missing_model_behavior(trainer_cls):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(reference_kl_enabled=False)
    assert trainer._reference_kl_values(0, None, None, None, 2, []) == []
    trainer.args.reference_kl_enabled = True
    with pytest.raises(RuntimeError, match="reference models are missing"):
        trainer._reference_kl_values(0, None, None, None, 2, [])


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer, MAGRPOTrainer])
@pytest.mark.parametrize("response_lens", [[], [2], [2, 1, 3]])
def test_sequence_kl_keeps_order_shortest_input_and_module_hook(
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


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer, MAGRPOTrainer])
def test_sequence_kl_restores_training_modes_and_rng(trainer_cls):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(reference_kl_enabled=True)
    policy, reference = _tiny_model(), _tiny_model()
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
