---
title: Multi-Turn Training
linkTitle: Multi-Turn Training
weight: 4
math: true
---

Many complex problems cannot be solved in a single turn. Agents need to interact with the environment to obtain useful feedback from other models or tools involved in the system, enabling iterative refinement and exploration of multiple solution paths.

## Multi-Turn MAGRPO

MAGRPO in the multi-turn setting (**MAGRPO-MT**) forms a tree-structured rollout expansion where branches represent different joint responses ([TreeRPO](https://arxiv.org/abs/2506.05183)). In each episode, a task is sampled from the dataset to construct initial observations and histories for all agents. At each turn, agents generate a group of joint responses from their current observation-action history, with each response initiating a distinct rollout. Agents receive joint rewards for each response based on the accumulated history and current action. **Each rollout then evolves independently**, producing new joint observations as the environment dynamics unfold and spawning more rollouts at the next turn. This process continues until the terminal turn is reached.

### Joint Mode

MAGRPO supports two modes for forming joint responses at each turn:

- **Align**: Provides flexibility in the number of joint responses generated per turn, allowing any number of generations at each turn. However, generations are not fully utilized since only aligned responses across agents are combined. As training progresses over {{< katex inline=true >}}T{{< /katex >}} turns with {{< katex inline=true >}}N{{< /katex >}} agents, the total number of leaves grows as {{< katex inline=true >}}G^T{{< /katex >}}, where {{< katex inline=true >}}G{{< /katex >}} is the number of generations per turn.

- **Cross**: Maximizes the utilization of generations and provides more accurate value estimation with more samples by forming the Cartesian product of all agent responses. As training progresses over {{< katex inline=true >}}T{{< /katex >}} turns with {{< katex inline=true >}}N{{< /katex >}} agents, the total number of leaves grows as {{< katex inline=true >}}G^{N \cdot T}{{< /katex >}}, where each node has {{< katex inline=true >}}G^N{{< /katex >}} sibling joint actions.

{{% hint warning %}}
Note that only responses originating from the same rollout can be combined, as rollouts evolve independently.
{{% /hint %}}

## External Feedback

External feedback mechanisms control how environment observations are incorporated into prompts for subsequent turns. CoMLRL provides built-in external transition modes as examples, and users can define custom external feedback functions to suit their specific tasks.

### Built-in Modes

CoMLRL's downstreaming environment repository  includes three example external transition modes for code generation tasks:

{{% hint success %}}

- **expert_edits**: Uses an external LLM (default: DeepSeek-Coder) to propose code edits. The follow-up prompts include edit suggestions with context from previous generations. This mode is configurable via `expert_model` to use different models (e.g., Claude, GPT) when API keys are available.

- **level_feedback**: Executes code against test cases and includes diagnostic feedback in the prompts. By default, includes the first test assertion; configurable via `sandbox_slice` to include all tests (0, None, or 'all'), specific number of tests, or last assertions (negative values).

- **plain**: Minimal feedback mode that includes previous responses and revision instructions without diagnostics or test results. Useful for tasks where external execution is not available or desired.
{{% /hint %}}

### Custom External Feedback

Users can implement custom external feedback by defining a function with the following interface:

{{% hint info %}}
Custom External Feedback Interface:

- `prompt`: Original task prompt/problem description (required)
- `agent_completions`: List or tuple of completions from the previous turn, one per agent (required)
- `num_agents`: Number of agents in the system (required)
- `prompt_history_per_agent`: List of prompt histories for each agent, where each history is a list of prompts from previous turns (optional)
- `response_history_per_agent`: List of response histories for each agent, where each history is a list of responses from previous turns (optional)
- `**kwargs`: Additional mode-specific parameters (optional)

The function should return a list or tuple of formatted prompts for the next turn, one for each agent.
{{% /hint %}}

For example:

```python
def custom_external(
    prompt: str,
    agent_completions: List[str],
    num_agents: int,
    prompt_history_per_agent: Optional[List[List[str]]] = None,
    response_history_per_agent: Optional[List[List[str]]] = None,
    **kwargs
) -> List[str]:
    # Custom logic to format next-turn prompts
    # Access environment feedback, tool outputs, etc.
    next_turn_prompts = []
    for i in range(num_agents):
        # Format prompt for agent i based on history and feedback
        next_prompt = f"{prompt}\nPrevious attempt: {agent_completions[i]}\nPlease revise."
        next_turn_prompts.append(next_prompt)
    return next_turn_prompts
```

This interface allows full flexibility in how environment feedback, tool outputs, or other contextual information is integrated into the multi-turn training loop.
