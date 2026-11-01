"""Config dataclasses validate their fields when constructed."""

import pytest

from comlrl.trainers.actor_critic.iac import IACConfig
from comlrl.trainers.actor_critic.maac import MAACConfig
from comlrl.trainers.pbrl.madpo import MADPOConfig
from comlrl.trainers.pbrl.marlhf import MARLHFConfig
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig

ACTOR_CRITIC = (IACConfig, MAACConfig)


def _case_id(config_cls, options):
    fields = ",".join(f"{key}={value!r}" for key, value in options.items())
    return f"{config_cls.__name__}({fields})"


def _invalid(config_cls, value, *fields, match=None, **options):
    """One case per field set to ``value``; the error must name the field."""
    return [
        pytest.param(
            config_cls,
            {field: value, **options},
            match or field,
            id=_case_id(config_cls, {field: value, **options}),
        )
        for field in fields
    ]


INVALID = [
    *[
        case
        for config_cls in ACTOR_CRITIC
        for case in [
            *_invalid(
                config_cls,
                0,
                "rollout_buffer_size",
                "train_batch_size",
                "num_agents",
                "num_turns",
                "eval_batch_size",
                "logging_steps",
            ),
            # Single-turn, so the lower bound is checked rather than the multi-turn rule.
            *_invalid(
                config_cls,
                0,
                "num_generations",
                match="num_generations must be >= 1",
                num_turns=1,
            ),
            *_invalid(config_cls, -1, "eval_interval", "eval_num_samples"),
            *_invalid(config_cls, "x", "critic_type"),
            *_invalid(config_cls, "invalid", "parallel_training"),
            *_invalid(config_cls, "auto", "parallel_training"),
            *_invalid(
                config_cls,
                "mp",
                "parallel_training",
                match="agent_devices",
                critic_devices="cpu",
            ),
            *_invalid(
                config_cls,
                "mp",
                "parallel_training",
                match="critic_devices",
                agent_devices="cpu",
            ),
            *_invalid(config_cls, 2, "num_generations", num_turns=2),
        ]
    ],
    *_invalid(
        MAGRPOConfig,
        0,
        "num_train_epochs",
        "num_agents",
        "rollout_buffer_size",
        "eval_batch_size",
        "num_turns",
        "logging_steps",
        "train_batch_size",
    ),
    *_invalid(MAGRPOConfig, -1, "eval_interval", "eval_num_samples"),
    *_invalid(MAGRPOConfig, 1, "num_generations"),
    *_invalid(MAGRPOConfig, "invalid", "parallel_training"),
    *_invalid(MAGRPOConfig, "auto", "parallel_training"),
    *_invalid(MAGRPOConfig, "mp", "parallel_training", match="agent_devices"),
    *_invalid(MADPOConfig, 2, "num_turns"),
    *_invalid(MADPOConfig, 1, "preference_num_candidates"),
    *_invalid(
        MADPOConfig,
        0,
        "preference_pairs_per_sample",
        "dpo_beta",
        "environment_steps_per_pair",
    ),
    *_invalid(MADPOConfig, "x", "pair_selection"),
    *_invalid(MARLHFConfig, "x", "rl_algorithm"),
    *_invalid(
        MARLHFConfig,
        0,
        "reward_learning_rate",
        "reward_num_train_epochs",
        "reward_train_batch_size",
        "reward_max_length",
    ),
]

VALID = [
    *[
        (config_cls, options)
        for config_cls in ACTOR_CRITIC
        for options in [
            {},
            {"parallel_training": "null"},
            {
                "parallel_training": "mp",
                "agent_devices": "cpu",
                "critic_devices": "cpu",
            },
            {"agent_devices": "cpu", "critic_devices": "cpu"},
            {"critic_type": "q"},
        ]
    ],
    *[
        (MAGRPOConfig, options)
        for options in [
            {},
            {"parallel_training": "null"},
            {"parallel_training": "mp", "agent_devices": "cpu"},
            {"agent_devices": "cpu"},
            {"num_generations": 2},
        ]
    ],
    *[
        (MADPOConfig, {"agent_devices": "cpu", **options})
        for options in [
            {},
            {"preference_pairs_per_sample": None},
            {"pair_selection": "random"},
        ]
    ],
    *[
        (MARLHFConfig, {"agent_devices": "cpu", **options})
        for options in [
            {},
            {"rl_algorithm": "maac"},
            {"reward_model_device": "cpu"},
        ]
    ],
]


@pytest.mark.parametrize("config_cls, options, match", INVALID)
def test_config_rejects_invalid_field(config_cls, options, match):
    with pytest.raises(ValueError, match=match):
        config_cls(**options)


@pytest.mark.parametrize(
    "config_cls, options",
    [pytest.param(*case, id=_case_id(*case)) for case in VALID],
)
def test_config_accepts_valid_options(config_cls, options):
    config_cls(**options)
