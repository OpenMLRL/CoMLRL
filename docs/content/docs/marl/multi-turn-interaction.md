---
title: Multi-Turn Interaction
weight: 4
math: true
tocDepth: 3
aliases:
  - /docs/multi-turn/
  - /docs/multi-turn/algorithm/
  - /docs/multi-turn/transition/
  - /docs/user-guide/multi-turn-training/
---

Many complex problems require more than one turn to solve. LLM agents interact with the environment to obtain useful feedback from other models or tools in the system.

## Multi-Turn MAGRPO

Multi-turn MAGRPO builds a rollout tree whose branches represent different joint responses ([TreeRPO](https://arxiv.org/abs/2506.05183)).

{{< rollout-diagram diagram="joint-tree" title="Joint rollout tree" description="Joint observations and actions branch into independent rollouts." >}}

Each episode begins with a task sampled from the dataset, which provides the initial observations {{< katex inline=true >}}\mathbf{o}_0=\{o_{1, 0}, \cdots, o_{n, 0}\}{{< /katex >}} and histories {{< katex inline=true >}}\mathbf{h}_0=\{h_{1, 0}, \cdots, h_{n, 0}\}{{< /katex >}} for all agents. At each turn, agents generate a group of joint responses {{< katex inline=true >}}\mathbf{a}^{\mathcal{G}}_t\gets\boldsymbol{\pi}^{\mathcal{G}}(\cdot|\mathbf{h}_t){{< /katex >}} from their current observation-action history {{< katex inline=true >}}\mathbf{h}_t{{< /katex >}}. Each response starts a distinct rollout. Agents receive joint rewards {{< katex inline=true >}}r^{(g)}_{t}{{< /katex >}} for each response, based on the accumulated history and current joint action {{< katex inline=true >}}\mathbf{a}^{(g)}_{t} \in \mathbf{a}^{\mathcal{G}}_{t}{{< /katex >}}. **Each rollout then evolves independently**, producing new joint observations {{< katex inline=true >}}\mathbf{o}^{\mathcal{G}}_{t+1}{{< /katex >}} as the environment changes and branching into further rollouts at the next turn {{< katex inline=true >}}t+1{{< /katex >}}. This process continues until the terminal turn {{< katex inline=true >}}H{{< /katex >}}.

### Joint Mode

MAGRPO supports two modes for forming joint responses at each turn:

- **Aligned** (`joint_mode="aligned"`): Responses with the same generation index are combined across agents. Each agent samples {{< katex inline=true >}}G{{< /katex >}} responses (the fixed `num_generations` value) at every node. With {{< katex inline=true >}}T{{< /katex >}} turns and no early termination, the tree has {{< katex inline=true >}}G^T{{< /katex >}} leaves.

{{< rollout-diagram diagram="align-tree" title="Aligned joint responses" description="Responses with matching generation indices are combined across agents." >}}

- **Cross** (`joint_mode="cross"`): The Cartesian product of the agents' sampled responses yields {{< katex inline=true >}}G^N{{< /katex >}} joint actions per node. With no early termination, the tree has {{< katex inline=true >}}G^{N \cdot T}{{< /katex >}} leaves. Responses are reused across combinations, so the joint samples are not independent. This increases the number of reward evaluations and the rollout size without guaranteeing more accurate value estimates.

{{< rollout-diagram diagram="cross-tree" title="Cross joint responses" description="The Cartesian product forms joint responses within each rollout." >}}

{{% hint warning %}}
Only responses from the same rollout can be combined because rollouts evolve independently.
The current MAGRPO implementation accumulates undiscounted returns: `discount` does not change its tree recursion. [MAREINFORCE]({{< relref "docs/marl/multi-agent-reinforce" >}}) describes return aggregation and advantage normalization.
{{% /hint %}}

## Multi-Turn MAAC/IAC

The MAGRPO rollout tree generates numerous samples per episode. In the multi-turn setting, MAAC/IAC generate only one sample per agent per turn, and the episode continues across turns until termination.
MAAC/IAC use one-step temporal-difference targets, with `discount` applied to the next rollout value and no bootstrapping after the final or early-terminated turn. The current implementation collects the full episode before buffering its samples. Policy updates occur when `rollout_buffer_size` is reached, and any remaining buffer is flushed at the end of the epoch. Policies are not updated between turns of the same episode.

{{< rollout-diagram diagram="ac" direction="right" title="Actor-critic trajectory" description="One joint response per turn advances the episode from left to right." >}}

{{% hint warning %}}
IAC/MAAC multi-turn training requires `num_generations` to be set to 1.
{{% /hint %}}

## Environment Transition

External feedback mechanisms control how environment observations are incorporated into prompts for subsequent turns. These mechanisms are usually implemented in CoMLRL's downstream environments.

{{< rollout-diagram diagram="environment-transition" direction="right" title="Environment transition" description="At turn t, the agents' responses enter the external transition, which returns each agent's own observation for turn t+1." >}}

### Custom External Feedback

Users can implement custom external feedback by defining a function with the following interface:

{{% hint info %}}
Custom External Feedback Interface:

- `prompt`: Original task prompt or problem description (required)
- `agent_completions`: List or tuple of completions from the previous turn, one per agent (required)
- `num_agents`: Number of agents in the system (required)
- `prompt_history_per_agent`: List containing one list of previous prompts per agent (optional)
- `response_history_per_agent`: List containing one list of previous responses per agent (optional)

The function must return a list or tuple of prompts for the next turn, one for each agent.
The trainer passes only the arguments above, with no additional keyword arguments. Mode-specific parameters can be configured through a closure or `functools.partial`.
{{% /hint %}}

By default, each returned prompt becomes the new `prompt` field and is passed through the corresponding agent's formatter.
If `external_prompt_passthrough=True`, the returned prompts are used directly without reformatting.
In MAGRPO, the external transition is called for each rollout branch with that branch's histories.

The following example constructs prompts for the next turn:

```python
from typing import List, Optional

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

### Example Modes {#example-modes-expert-diagnosis-and-self-improvement}

An [environment for code generation](https://github.com/OpenMLRL/LLM_Collab_Code_Generation) includes three example external transition modes:

{{% hint success %}}

- `external.mode=expert_edits`: An external LLM (default: DeepSeek-Coder) proposes code edits. Follow-up prompts include edit suggestions and context from previous turns. With the required API keys, users can select a different expert (e.g., Claude, GPT) through `expert_model`.

- `external.mode=level_feedback`: Static AST checks and dynamic code execution provide a diagnosis. By default, the sandbox includes only the first test. The `sandbox_slice` parameter selects all tests (0, None, or 'all') or a specific number of tests (negative values are supported).

- `external.mode=plain`: This self-improvement mode includes only the prompts and responses from previous turns and a revision instruction.
{{% /hint %}}
