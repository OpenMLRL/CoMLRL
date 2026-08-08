import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import torch


RewardCallable = TypeVar("RewardCallable", bound=Callable[..., Any])


def resolve_reward_range(
    reward_func: Callable[..., Any],
) -> Optional[Tuple[float, float]]:
    """Return the raw reward range declared by a reward callable, if present."""
    reward_range = getattr(reward_func, "reward_range", None)
    if reward_range is None:
        return None
    if (
        isinstance(reward_range, (str, bytes))
        or not isinstance(reward_range, Sequence)
        or len(reward_range) != 2
    ):
        raise ValueError(
            "reward_func.reward_range must be a two-item sequence: "
            "(minimum, maximum)."
        )
    minimum, maximum = (float(value) for value in reward_range)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("reward_func.reward_range values must be finite.")
    if minimum >= maximum:
        raise ValueError("reward_func.reward_range minimum must be less than maximum.")
    return minimum, maximum


def set_reward_range(
    reward_func: RewardCallable,
    minimum: float,
    maximum: float,
) -> RewardCallable:
    """Declare the raw output range of a reward callable and return it."""
    setattr(reward_func, "reward_range", (float(minimum), float(maximum)))
    resolve_reward_range(reward_func)
    return reward_func


def call_reward_function(
    reward_func,
    prompts: Sequence[str],
    agent_completions: Sequence[Sequence[str]],
    *,
    num_agents: int,
    batch_items: Optional[Sequence[Dict[str, Any]]] = None,
    signature: Optional[inspect.Signature] = None,
) -> List[float]:
    if signature is None:
        try:
            signature = inspect.signature(reward_func)
        except (TypeError, ValueError):
            signature = None
    if signature is None:

        def _coerce(raw):
            if isinstance(raw, torch.Tensor):
                rewards = raw.detach().cpu().tolist()
            elif isinstance(raw, (list, tuple)):
                rewards = list(raw)
            else:
                rewards = [float(raw)]
            return [float(r) for r in rewards]

        if batch_items is not None:
            try:
                return _coerce(reward_func(*agent_completions, batch_items=batch_items))
            except TypeError:
                pass
        try:
            return _coerce(reward_func(*agent_completions))
        except TypeError:
            return _coerce(reward_func(agent_completions))

    params = signature.parameters
    param_list = list(params.values())
    positional_params = [
        p
        for p in param_list
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    param_count = len(positional_params)
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in param_list)
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in param_list)
    has_batch_items = "batch_items" in params
    has_prompts_kw = "prompts" in params

    def _call_with_args():
        kwargs: Dict[str, Any] = {}
        if batch_items is not None and (has_batch_items or has_varkw):
            kwargs["batch_items"] = batch_items

        if has_varargs:
            args = list(agent_completions)
            used_prompts = False
        else:
            if num_agents == 1:
                if param_count == 1:
                    args = [agent_completions[0]]
                    used_prompts = False
                else:
                    args = [prompts, agent_completions[0]]
                    used_prompts = True
            else:
                if param_count == num_agents:
                    args = list(agent_completions)
                    used_prompts = False
                elif param_count == num_agents + 1:
                    args = [prompts] + list(agent_completions)
                    used_prompts = True
                elif param_count == 1:
                    args = [agent_completions]
                    used_prompts = False
                else:
                    args = list(agent_completions)
                    used_prompts = False

        if not used_prompts and (has_prompts_kw or has_varkw):
            kwargs["prompts"] = prompts

        return reward_func(*args, **kwargs)  # type: ignore[arg-type]

    def _coerce(raw):
        if isinstance(raw, torch.Tensor):
            rewards = raw.detach().cpu().tolist()
        elif isinstance(raw, (list, tuple)):
            rewards = list(raw)
        else:
            rewards = [float(raw)]
        return [float(r) for r in rewards]

    try:
        return _coerce(_call_with_args())
    except TypeError as exc:
        try:
            return _coerce(reward_func(*agent_completions))  # type: ignore[arg-type]
        except TypeError:
            raise exc


def normalize_reward_lengths(
    rewards: Sequence[float],
    *,
    num_agents: int,
    num_generations: int,
    algorithm: str,
) -> List[float]:
    processed = [float(r) for r in rewards]
    if num_agents == 1:
        return processed

    if num_generations < 1:
        raise ValueError("num_generations must be >= 1.")

    if len(processed) == 1:
        return processed * num_agents
    if len(processed) == num_agents:
        return processed
    if len(processed) == num_generations:
        return processed
    raise ValueError(
        f"Reward function must return either 1 or {num_agents} values per prompt for multi-agent {algorithm}."
    )
