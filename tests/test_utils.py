"""Reward, formatter and W&B helpers in comlrl.utils."""

import inspect

import pytest
import torch

from comlrl.utils import RewardProcessors, build_formatters, call_reward_function
from comlrl.utils.wandb_utils import wandb_init_kwargs


@pytest.mark.parametrize("inspectable", [True, False])
@pytest.mark.parametrize(
    "raw, expected",
    [
        (2, [2.0]),
        ([1, -2], [1.0, -2.0]),
        ((1.5, 2.5), [1.5, 2.5]),
        ([], []),
        (torch.tensor([1.5, -2.0], requires_grad=True), [1.5, -2.0]),
    ],
)
def test_call_reward_function_normalizes_output_types(inspectable, raw, expected):
    calls = []

    class Reward:
        @property
        def __signature__(self):
            if not inspectable:
                raise ValueError("No signature")
            return inspect.signature(self.__call__)

        def __call__(self, first, second):
            calls.append((first, second))
            return raw

    completions = [["first"], ["second"]]
    assert (
        call_reward_function(Reward(), ["prompt"], completions, num_agents=2)
        == expected
    )
    assert calls == [(completions[0], completions[1])]


def test_call_reward_function_falls_back_in_order_without_signature():
    calls = []

    class Reward:
        @property
        def __signature__(self):
            raise ValueError("No signature")

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            if kwargs or len(args) != 1:
                raise TypeError("Expected one joint completion list")
            return (3,)

    completions = [["first"], ["second"]]
    items = [{"prompt": "prompt"}]
    result = call_reward_function(
        Reward(), ["prompt"], completions, num_agents=2, batch_items=items
    )
    assert result == [3.0]
    assert calls == [
        (tuple(completions), {"batch_items": items}),
        (tuple(completions), {}),
        ((completions,), {}),
    ]


def test_call_reward_function_reraises_original_type_error():
    calls = []
    original_error = TypeError("Reward rejected batch metadata")

    def reward(*completions, **kwargs):
        calls.append(kwargs)
        if kwargs:
            raise original_error
        raise TypeError("Fallback failed")

    items = [{"prompt": "prompt"}]
    with pytest.raises(TypeError) as exc:
        call_reward_function(
            reward,
            ["prompt"],
            [["first"], ["second"]],
            num_agents=2,
            batch_items=items,
        )
    assert exc.value is original_error
    assert calls == [{"batch_items": items, "prompts": ["prompt"]}, {}]


def test_reward_processors_scale_and_shift():
    assert RewardProcessors.scale(factor=100.0)(0.25) == 25.0
    assert RewardProcessors.shift(value=-1.0)(0.25) == -0.75
    assert RewardProcessors.scale()(3.0) == RewardProcessors.shift()(3.0) == 3.0


@pytest.mark.parametrize("pass_none", [True, False])
def test_build_formatters_default_and_external_prompts(pass_none):
    item = {"prompt": "original"}
    default = build_formatters(None, 2, pass_none_external_prompts=pass_none)
    assert [formatter(item) for formatter in default] == ["original", "original"]
    assert default[0]({}, external_prompts="external") == "external"
    assert default[0](item, external_prompts="") == ""
    assert default[0]({}) == ""

    seen = []

    def accepts_external(value, external_prompts="default-argument"):
        seen.append((value, external_prompts))
        return external_prompts

    wrapped = build_formatters(
        accepts_external, 1, pass_none_external_prompts=pass_none
    )[0]
    expected_default = None if pass_none else "default-argument"
    assert wrapped(item) == expected_default
    assert wrapped(item, external_prompts="external") == "external"
    assert seen == [(item, expected_default), (item, "external")]
    single_argument = build_formatters(lambda value: value["prompt"], 1)[0]
    assert single_argument(item, external_prompts="ignored") == "original"


def test_wandb_init_kwargs_defaults():
    assert wandb_init_kwargs({"project": "p"}, {}) == {
        "project": "p",
        "entity": None,
        "name": None,
        "config": {},
    }


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
