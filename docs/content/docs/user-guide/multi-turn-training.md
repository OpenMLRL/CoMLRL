---
title: Multi-Turn Interaction
linkTitle: Multi-Turn Interaction
weight: 3
math: true
---

Many complex problems cannot be solved in a single turn. LLM agents need to interact with the environment to obtain useful feedback from other models or tools involved in the system.

## Multi-Turn MAGRPO

MAGRPO in the multi-turn setting forms a tree-structured rollout expansion where branches represent different joint responses ([TreeRPO](https://arxiv.org/abs/2506.05183)).

{{< rollout-diagram diagram="joint-tree" title="Joint rollout tree" description="Joint observations and actions branch into independent rollouts." >}}

In each episode, a task is sampled from the dataset to construct initial observations {{< katex inline=true >}}\mathbf{o}_0=\{o_{1, 0}, \cdots, o_{n, 0}\}{{< /katex >}} and histories {{< katex inline=true >}}\mathbf{h}_0=\{h_{1, 0}, \cdots, h_{n, 0}\}{{< /katex >}} for all agents. At each turn, agents generate a group of joint responses {{< katex inline=true >}}\mathbf{a}^{\mathcal{G}}_t\gets\boldsymbol{\pi}^{\mathcal{G}}(\cdot|\mathbf{h}_t){{< /katex >}} from their current observation-action history {{< katex inline=true >}}\mathbf{h}_t{{< /katex >}}, with each response initiating a distinct rollout. Agents receive joint rewards {{< katex inline=true >}}r^{(g)}_{t}{{< /katex >}} for each response based on the accumulated history {{< katex inline=true >}}\mathbf{a}^{(g)}_{t} \in \mathbf{a}^{\mathcal{G}}_{t}{{< /katex >}} and current action. **Each rollout then evolves independently**, producing new joint observations {{< katex inline=true >}}\mathbf{o}^{\mathcal{G}}_{t+1}{{< /katex >}} as the environment dynamics unfold and spawning more rollouts at the next turn {{< katex inline=true >}}t+1{{< /katex >}}. This process continues until the terminal turn is reached {{< katex inline=true >}}H{{< /katex >}}.

### Joint Mode

MAGRPO supports two modes for forming joint responses at each turn:

- **Aligned** (`joint_mode="aligned"`): Combines the responses with the same generation index across agents. Each agent samples the fixed `num_generations` value {{< katex inline=true >}}G{{< /katex >}} at every node. With {{< katex inline=true >}}T{{< /katex >}} turns and no early termination, the tree has {{< katex inline=true >}}G^T{{< /katex >}} leaves.

{{< rollout-diagram diagram="align-tree" title="Aligned joint responses" description="Responses with matching generation indices are combined across agents." >}}

- **Cross** (`joint_mode="cross"`): Evaluates the Cartesian product of the agents' sampled responses, yielding {{< katex inline=true >}}G^N{{< /katex >}} joint actions per node. With no early termination, the tree has {{< katex inline=true >}}G^{N \cdot T}{{< /katex >}} leaves. Responses are reused across combinations, so these are not independent joint samples. This increases reward evaluations and rollout size; it does not guarantee more accurate value estimates.

{{< rollout-diagram diagram="cross-tree" title="Cross joint responses" description="The Cartesian product forms joint responses within each rollout." >}}

{{% hint warning %}}
Note that only responses originating from the same rollout can be combined, as rollouts evolve independently.
The current MAGRPO implementation accumulates undiscounted returns: `discount` does not change its tree recursion. See [MA-REINFORCE]({{< relref "docs/marl/multi-agent-reinforce" >}}) for return aggregation and advantage normalization.
{{% /hint %}}

## Multi-Turn MAAC/IAC

Different from MAGRPO rollout tree that generates numerous samples in each episode, MAAC/IAC with multi-turn training only generates one sample per agent at each turn in the multi-turning setting, and the episode continues for multiple turns until termination.
MAAC/IAC use one-step temporal-difference targets, with `discount` applied to the next rollout value and no bootstrap after the final or early-terminated turn. The current implementation collects the full episode first, then buffers its samples and updates when `rollout_buffer_size` is reached (or flushes the remaining buffer at epoch end). It does not update policies between turns of the same episode.
The external feedback mechanism controls how environment observations are incorporated into prompts for subsequent turns, which is usually implemented in CoMLRL's downstreaming environments.

{{< rollout-diagram diagram="ac" direction="right" title="Actor-critic trajectory" description="One joint response per turn advances the episode from left to right." >}}

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

<div id="centralized-preference-collaboration"></div>
<div id="api-comparators"></div>

{{% hint info %}}
For actor/comparator layouts and centralized preference training, see [Training Paradigms]({{< relref "/docs/mapl/training-paradigms#centralized-preference-collaboration" >}}).
Remote comparator configuration is covered in [Comparator]({{< relref "/docs/mapl/comparator#api-comparators" >}}).
{{% /hint %}}
