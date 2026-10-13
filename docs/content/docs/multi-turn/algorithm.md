---
title: Algorithm Property
weight: 1
math: true
aliases:
  - /docs/user-guide/multi-turn-training/
---

Many complex problems cannot be solved in a single turn. LLM agents need to interact with the environment to obtain useful feedback from other models or tools involved in the system.

## Multi-Turn MAGRPO

MAGRPO in the multi-turn setting forms a tree-structured rollout expansion where branches represent different joint responses ([TreeRPO](https://arxiv.org/abs/2506.05183)).

{{< rollout-diagram diagram="joint-tree" title="Joint rollout tree" description="Joint observations and actions branch into independent rollouts." >}}

In each episode, a task is sampled from the dataset to construct initial observations {{< katex inline=true >}}\mathbf{o}_0=\{o_{1, 0}, \cdots, o_{n, 0}\}{{< /katex >}} and histories {{< katex inline=true >}}\mathbf{h}_0=\{h_{1, 0}, \cdots, h_{n, 0}\}{{< /katex >}} for all agents. At each turn, agents generate a group of joint responses {{< katex inline=true >}}\mathbf{a}^{\mathcal{G}}_t\gets\boldsymbol{\pi}^{\mathcal{G}}(\cdot|\mathbf{h}_t){{< /katex >}} from their current observation-action history {{< katex inline=true >}}\mathbf{h}_t{{< /katex >}}, with each response initiating a distinct rollout. Agents receive joint rewards {{< katex inline=true >}}r^{(g)}_{t}{{< /katex >}} for each response based on the accumulated history {{< katex inline=true >}}\mathbf{a}^{(g)}_{t} \in \mathbf{a}^{\mathcal{G}}_{t}{{< /katex >}} and current action. **Each rollout then evolves independently**, producing new joint observations {{< katex inline=true >}}\mathbf{o}^{\mathcal{G}}_{t+1}{{< /katex >}} as the environment dynamics unfold and spawning more rollouts at the next turn {{< katex inline=true >}}t+1{{< /katex >}}. This process continues until the terminal turn {{< katex inline=true >}}H{{< /katex >}} is reached.

### Joint Mode

MAGRPO supports two modes for forming joint responses at each turn:

- **Aligned** (`joint_mode="aligned"`): Combines the responses with the same generation index across agents. Each agent samples {{< katex inline=true >}}G{{< /katex >}} responses (the fixed `num_generations` value) at every node. With {{< katex inline=true >}}T{{< /katex >}} turns and no early termination, the tree has {{< katex inline=true >}}G^T{{< /katex >}} leaves.

{{< rollout-diagram diagram="align-tree" title="Aligned joint responses" description="Responses with matching generation indices are combined across agents." >}}

- **Cross** (`joint_mode="cross"`): Evaluates the Cartesian product of the agents' sampled responses, yielding {{< katex inline=true >}}G^N{{< /katex >}} joint actions per node. With no early termination, the tree has {{< katex inline=true >}}G^{N \cdot T}{{< /katex >}} leaves. Responses are reused across combinations, so these are not independent joint samples. This increases reward evaluations and rollout size; it does not guarantee more accurate value estimates.

{{< rollout-diagram diagram="cross-tree" title="Cross joint responses" description="The Cartesian product forms joint responses within each rollout." >}}

{{% hint warning %}}
Note that only responses originating from the same rollout can be combined, as rollouts evolve independently.
The current MAGRPO implementation accumulates undiscounted returns: `discount` does not change its tree recursion. See [MAREINFORCE]({{< relref "docs/marl/multi-agent-reinforce" >}}) for return aggregation and advantage normalization.
{{% /hint %}}

## Multi-Turn MAAC/IAC

Unlike the MAGRPO rollout tree, which generates numerous samples in each episode, MAAC/IAC generate only one sample per agent at each turn in the multi-turn setting, and the episode continues for multiple turns until termination.
MAAC/IAC use one-step temporal-difference targets, with `discount` applied to the next rollout value and no bootstrap after the final or early-terminated turn. The current implementation collects the full episode first, then buffers its samples and updates when `rollout_buffer_size` is reached (or flushes the remaining buffer at epoch end). It does not update policies between turns of the same episode.
The external feedback mechanism controls how environment observations are incorporated into prompts for subsequent turns; it is usually implemented in CoMLRL's downstream environments.

{{< rollout-diagram diagram="ac" direction="right" title="Actor-critic trajectory" description="One joint response per turn advances the episode from left to right." >}}
