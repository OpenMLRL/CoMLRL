---
title: Multi-Turn Interaction
linkTitle: Multi-Turn Interaction
weight: 5
math: true
---

Many complex problems cannot be solved in a single turn. LLM agents need to interact with the environment to obtain useful feedback from other models or tools involved in the system.

## Multi-Turn MAGRPO

MAGRPO in the multi-turn setting forms a tree-structured rollout expansion where branches represent different joint responses ([TreeRPO](https://arxiv.org/abs/2506.05183)).

<p align="center">
    <img src="/img/joint-tree.svg" width="450px;" alt=""/>
</p>

In each episode, a task is sampled from the dataset to construct initial observations {{< katex inline=true >}}\mathbf{o}_0=\{o_{1, 0}, \cdots, o_{n, 0}\}{{< /katex >}} and histories {{< katex inline=true >}}\mathbf{h}_0=\{h_{1, 0}, \cdots, h_{n, 0}\}{{< /katex >}} for all agents. At each turn, agents generate a group of joint responses {{< katex inline=true >}}\mathbf{a}^{\mathcal{G}}_t\gets\boldsymbol{\pi}^{\mathcal{G}}(\cdot|\mathbf{h}_t){{< /katex >}} from their current observation-action history {{< katex inline=true >}}\mathbf{h}_t{{< /katex >}}, with each response initiating a distinct rollout. Agents receive joint rewards {{< katex inline=true >}}r^{(g)}_{t}{{< /katex >}} for each response based on the accumulated history {{< katex inline=true >}}\mathbf{a}^{(g)}_{t} \in \mathbf{a}^{\mathcal{G}}_{t}{{< /katex >}} and current action. **Each rollout then evolves independently**, producing new joint observations {{< katex inline=true >}}\mathbf{o}^{\mathcal{G}}_{t+1}{{< /katex >}} as the environment dynamics unfold and spawning more rollouts at the next turn {{< katex inline=true >}}t+1{{< /katex >}}. This process continues until the terminal turn is reached {{< katex inline=true >}}H{{< /katex >}}.

### Joint Mode

MAGRPO supports two modes for forming joint responses at each turn:

- **Align**: Provides flexibility in the number of joint responses generated per turn, allowing any number of generations at each turn. However, generations are not fully utilized since only aligned responses across agents are combined. As training progresses over {{< katex inline=true >}}T{{< /katex >}} turns with {{< katex inline=true >}}N{{< /katex >}} agents, the total number of leaves grows as {{< katex inline=true >}}G^T{{< /katex >}}, where {{< katex inline=true >}}G{{< /katex >}} is the number of generations per turn.

<p align="center">
    <img src="/img/align-tree.svg" width="500px;" alt=""/>
</p>

- **Cross**: Maximizes the utilization of generations and provides more accurate value estimation with more samples by forming the Cartesian product of all agent responses. As training progresses over {{< katex inline=true >}}T{{< /katex >}} turns with {{< katex inline=true >}}N{{< /katex >}} agents, the total number of leaves grows as {{< katex inline=true >}}G^{N \cdot T}{{< /katex >}}, where each node has {{< katex inline=true >}}G^N{{< /katex >}} sibling joint actions.

<p align="center">
    <img src="/img/cross-tree.svg" width="500px;" alt=""/>
</p>

{{% hint warning %}}
Note that only responses originating from the same rollout can be combined, as rollouts evolve independently.
{{% /hint %}}

## Multi-Turn MAAC/IAC

Different from MAGRPO rollout tree that generates numerous samples in each episode, MAAC/IAC with multi-turn training only generates one sample per agent at each turn in the multi-turning setting, and the episode continues for multiple turns until termination.
Since the value estimation in MAAC/IAC is updated based on temporal difference error, the agents don't need to wait until the end of the episode to update their policies, and can learn online. This feature is especially useful in the coordination optimization in long-horizon settings.
The external feedback mechanism controls how environment observations are incorporated into prompts for subsequent turns, which is usually implemented in CoMLRL's downstreaming environments.

<p align="center">
    <img src="/img/ac.svg" width="480px;" alt=""/></br>
</p>

## Environment Transition

External feedback mechanisms control how environment observations are incorporated into prompts for subsequent turns, this is usually implemented in CoMLRL's downstreaming environments.

### Custom External Feedback

Users can implement custom external feedback by defining a function with the following interface:

{{% hint info %}}
Custom External Feedback Interface:

- `prompt`: Original task prompt/problem description (required)
- `agent_completions`: List or tuple of completions from the previous turn, one per agent (required)
- `num_agents`: Number of agents in the system (required)
- `prompt_history_per_agent`: List of prompt histories for each agent, where each history is a list of prompts from previous turns (optional)
- `response_history_per_agent`: List of response histories for each agent, where each history is a list of responses from previous turns (optional)

The function must return a list or tuple of prompts for the next turn, one for each agent.
The trainer only passes the arguments above (no extra kwargs), so any mode-specific parameters should be captured via closure or `functools.partial`.
{{% /hint %}}

By default, returned prompts are inserted as the new `prompt` field and then passed through each agent's formatter.
If `external_prompt_passthrough=true`, the returned prompts are used directly without re-formatting.
In MAGRPO, the external transition is called per rollout branch with that branch's histories.

For example:

```python
def custom_external(
    prompt: str,
    agent_completions: List[str],
    num_agents: int,
    prompt_history_per_agent: Optional[List[List[str]]] = None,
    response_history_per_agent: Optional[List[List[str]]] = None,
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

{{% hint warning %}}
For IAC/MAAC multi-turn training, `num_generations` must be set to 1.
{{% /hint %}}

### Example Modes (Expert, Diagnosis, and Self-Improvement)

An [environment for code generation](https://github.com/OpenMLRL/LLM_Collab_Code_Generation) includes 3 example external transition modes:

{{% hint success %}}

- `external.mode=expert_edits`: Uses an external LLM (default: DeepSeek-Coder) to propose code edits. Follow-up prompts include edit suggestions with context from previous turns. It can be configured via `expert_model` for different experts (e.g., Claude, GPT) when API keys are available.

- `external.mode=level_feedback`: Static AST checks and dynamically executes code to provide diagnosis. The default sandbox test includes the first test; configurable via `sandbox_slice` to include all tests (0, None, or 'all'), specific number of tests (negative values enabled).

- `external.mode=plain`: Self-improvement mode that just includes prompts and responses in the previous turns and a revision instruction.
{{% /hint %}}

## API Comparators

Iterative MADPO and MARLHF can use a remote model with `comparator_policy=api`.
The existing `comparator_api_format=openai` path remains compatible with
OpenAI-style Chat Completions endpoints such as DeepSeek. Two native formats
are also supported:

- `anthropic`: Anthropic Messages API. Use `https://api.anthropic.com/v1/messages`,
  `ANTHROPIC_API_KEY`, and a model such as `claude-sonnet-4-20250514`.
- `openai_responses` (or `responses`/`codex`): OpenAI Responses API. Use
  `https://api.openai.com/v1/responses`, `OPENAI_API_KEY`, and a model such as
  `gpt-5.1-codex`.

The native APIs are queried once per candidate because they do not use the
Chat Completions `n` parameter. The prompt remains the same centralized or
decentralized comparator prompt, and the trainer extracts text from the
provider-specific response format.

Example Anthropic comparator overrides:

```text
comparator_policy=api
comparator_api_format=anthropic
comparator_api_url=https://api.anthropic.com/v1/messages
comparator_api_model=claude-sonnet-4-20250514
comparator_api_key_env=ANTHROPIC_API_KEY
```

Example OpenAI/Codex comparator overrides:

```text
comparator_policy=api
comparator_api_format=openai_responses
comparator_api_url=https://api.openai.com/v1/responses
comparator_api_model=gpt-5.1-codex
comparator_api_key_env=OPENAI_API_KEY
```
