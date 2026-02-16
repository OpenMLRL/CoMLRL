import pytest

from comlrl.trainers.actor_critic.iac import IACConfig
from comlrl.trainers.actor_critic.maac import MAACConfig
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig


def _assert_invalid(cfg_cls, field, value, match=None):
    with pytest.raises(ValueError, match=match or field):
        cfg_cls(**{field: value})


def _assert_invalid_fields(cfg_cls, fields, value):
    for field in fields:
        _assert_invalid(cfg_cls, field, value)


def test_iac_config_constraints():
    _assert_invalid_fields(
        IACConfig,
        [
            "rollout_buffer_size",
            "train_batch_size",
            "num_agents",
            "num_turns",
            "eval_batch_size",
            "logging_steps",
        ],
        0,
    )
    _assert_invalid_fields(IACConfig, ["eval_interval", "eval_num_samples"], -1)
    _assert_invalid(IACConfig, "num_generations", 0)
    _assert_invalid(IACConfig, "critic_type", "x")
    _assert_invalid(IACConfig, "parallel_training", "invalid")
    _assert_invalid(IACConfig, "parallel_training", "auto")
    with pytest.raises(ValueError, match="agent_devices"):
        IACConfig(critic_devices="cpu")
    with pytest.raises(ValueError, match="critic_devices"):
        IACConfig(agent_devices="cpu")
    with pytest.raises(ValueError, match="num_generations"):
        IACConfig(
            num_turns=2,
            num_generations=2,
            agent_devices="cpu",
            critic_devices="cpu",
        )

    IACConfig(agent_devices="cpu", critic_devices="cpu")
    IACConfig(
        num_turns=2,
        num_generations=1,
        agent_devices="cpu",
        critic_devices="cpu",
    )
    IACConfig(critic_type="q", agent_devices="cpu", critic_devices="cpu")


def test_maac_config_constraints():
    _assert_invalid_fields(
        MAACConfig,
        [
            "rollout_buffer_size",
            "train_batch_size",
            "num_agents",
            "num_generations",
            "num_turns",
            "eval_batch_size",
            "logging_steps",
        ],
        0,
    )
    _assert_invalid_fields(MAACConfig, ["eval_interval", "eval_num_samples"], -1)
    _assert_invalid(MAACConfig, "critic_type", "x")
    _assert_invalid(MAACConfig, "parallel_training", "invalid")
    _assert_invalid(MAACConfig, "parallel_training", "auto")
    with pytest.raises(ValueError, match="agent_devices"):
        MAACConfig(critic_devices="cpu")
    with pytest.raises(ValueError, match="critic_devices"):
        MAACConfig(agent_devices="cpu")
    with pytest.raises(ValueError, match="num_generations"):
        MAACConfig(
            num_turns=2,
            num_generations=2,
            agent_devices="cpu",
            critic_devices="cpu",
        )

    MAACConfig(agent_devices="cpu", critic_devices="cpu")
    MAACConfig(
        num_turns=2,
        num_generations=1,
        agent_devices="cpu",
        critic_devices="cpu",
    )
    MAACConfig(critic_type="q", agent_devices="cpu", critic_devices="cpu")


def test_magrpo_config_constraints():
    _assert_invalid_fields(
        MAGRPOConfig,
        [
            "num_train_epochs",
            "num_agents",
            "rollout_buffer_size",
            "eval_batch_size",
            "num_turns",
            "logging_steps",
            "train_batch_size",
        ],
        0,
    )
    _assert_invalid_fields(MAGRPOConfig, ["eval_interval", "eval_num_samples"], -1)
    _assert_invalid(MAGRPOConfig, "num_generations", 1)
    _assert_invalid(MAGRPOConfig, "parallel_training", "invalid")
    _assert_invalid(MAGRPOConfig, "parallel_training", "auto")
    with pytest.raises(ValueError, match="agent_devices"):
        MAGRPOConfig()

    MAGRPOConfig(agent_devices="cpu")
    MAGRPOConfig(num_generations=2, agent_devices="cpu")
