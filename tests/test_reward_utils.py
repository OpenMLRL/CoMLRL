import inspect

import pytest
import torch

from comlrl.utils.reward_utils import call_reward_function


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
def test_reward_output_types_preserve_values_and_callback_count(
    inspectable, raw, expected
):
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


def test_uninspectable_reward_keeps_fallback_order():
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


def test_inspectable_reward_reraises_original_type_error():
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
