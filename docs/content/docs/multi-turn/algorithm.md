---
title: Multi-Turn Algorithm
weight: 1
math: true
aliases:
  - /docs/user-guide/multi-turn-training/
---

Many complex problems require more than one turn to solve. LLM agents interact with the environment to obtain useful feedback from other models or tools in the system.

## Multi-Turn MAGRPO

Multi-turn MAGRPO builds a rollout tree whose branches represent different joint responses ([TreeRPO](https://arxiv.org/abs/2506.05183)).

{{< rollout-diagram diagram="joint-tree" title="Joint rollout tree" description="Joint observations and actions branch into independent rollouts." >}}

Each episode begins with a task sampled from the dataset, which provides the initial observations {{< katex inline=true >}}\mathbf{o}_0=\{o_{1, 0}, \cdots, o_{n, 0}\}{{< /katex >}} and histories {{< katex inline=true >}}\mathbf{h}_0=\{h_{1, 0}, \cdots, h_{n, 0}\}{{< /katex >}} for all agents. At each turn, agents generate a group of joint responses {{< katex inline=true >}}\mathbf{a}^{\mathcal{G}}_t\gets\boldsymbol{\pi}^{\mathcal{G}}(\cdot|\mathbf{h}_t){{< /katex >}} from their current observation-action history {{< katex inline=true >}}\mathbf{h}_t{{< /katex >}}. Each response starts a distinct rollout. Agents receive joint rewards {{< katex inline=true >}}r^{(g)}_{t}{{< /katex >}} for each response, based on the accumulated history and current joint action {{< katex inline=true >}}\mathbf{a}^{(g)}_{t} \in \mathbf{a}^{\mathcal{G}}_{t}{{< /katex >}}. **Each rollout then evolves independently**, producing new joint observations {{< katex inline=true >}}\mathbf{o}^{\mathcal{G}}_{t+1}{{< /katex >}} as the environment evolves and branching into further rollouts at the next turn {{< katex inline=true >}}t+1{{< /katex >}}. This process continues until the terminal turn {{< katex inline=true >}}H{{< /katex >}}.

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
External feedback controls how environment observations are incorporated into prompts for subsequent turns. This mechanism is usually implemented in CoMLRL's downstream environments.

{{< rollout-diagram diagram="ac" direction="right" title="Actor-critic trajectory" description="One joint response per turn advances the episode from left to right." >}}
