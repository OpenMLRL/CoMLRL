---
title: Multi-LLM PbRL
weight: 3
aliases:
  - /docs/user-guide/multi-agent-preference-learning/
math: true
---

Multi-Agent Preference Learning (MAPL) improves collaborating agents by comparing joint responses to the same task. A preference identifies a better and a worse joint response, so each agent learns in the context of the team's outcome.

CoMLRL provides two approaches: **MADPO** directly optimizes the policies from preference pairs, while **MARLHF** first learns a joint reward model and then trains the policies with that reward. Their iterative variants refresh the preference data as training progresses. MAPL names this family of methods; the public trainer classes are `MADPOTrainer`, `MARLHFTrainer`, `MADPOIterTrainer`, and `MARLHFIterTrainer`.

Start with the shared training framework, then the two learning approaches:

- [Training Framework]({{< relref "/docs/mapl/training-framework" >}}): Actor and comparator layouts, iterative training, comparator sources, and replay.
- [MARLHF]({{< relref "/docs/mapl/marlhf" >}}): Two-stage learning with a joint reward model followed by policy optimization.
- [MADPO]({{< relref "/docs/mapl/madpo" >}}): Direct policy optimization from joint preference pairs.

## Preference Pairs

For a task with per-agent prompts {{< katex >}}\mathbf{h}=(h_1,\ldots,h_N){{< /katex >}}, the trainer samples candidate joint responses {{< katex >}}\mathbf{a}^{k}=(a_1^k,\ldots,a_N^k){{< /katex >}}. Candidates with the same index are aligned across agents and receive a shared scalar task reward. A pair is ordered as {{< katex >}}\mathbf{a}^{+}\succ\mathbf{a}^{-}{{< /katex >}} when the first response has a strictly higher reward. Ties are discarded.

The training dataset contains task items used to construct prompts. By default, the formatter reads each item's `prompt` field; custom `formatters` can use other fields or provide different prompts for each role. The trainers generate and score their own preference pairs from these items. The base MADPO and MARLHF trainers collect those pairs once before optimization.

{{% hint warning %}}
All four preference trainers currently require `num_turns=1` and aligned joint responses (`joint_mode="aligned"`). The iterative variants repeat data collection and training; an iteration is not a dialog turn.
{{% /hint %}}

## MARLHF

Multi-Agent Reinforcement Learning from Human Feedback uses two stages: first fit a joint reward model to preferred and rejected team responses, then optimize the policies with that learned reward. See [MARLHF]({{< relref "/docs/mapl/marlhf" >}}) for the reward objective and policy-training workflow.

<span id="api-marlhfconfig"></span>
<span id="api-marlhftrainer"></span>

The [MARLHFConfig parameters]({{< relref "/docs/mapl/marlhf#api-marlhfconfig" >}}) and [MARLHFTrainer setup]({{< relref "/docs/mapl/marlhf#api-marlhftrainer" >}}) are documented on that page.

## MADPO

Multi-Agent Direct Preference Optimization updates the policies directly from preferred and rejected joint responses. See [MADPO]({{< relref "/docs/mapl/madpo" >}}) for its joint preference loss and training workflow.

<span id="api-madpoconfig"></span>
<span id="api-madpotrainer"></span>

The [MADPOConfig parameters]({{< relref "/docs/mapl/madpo#api-madpoconfig" >}}) and [MADPOTrainer setup]({{< relref "/docs/mapl/madpo#api-madpotrainer" >}}) are documented on that page.

## Iterative Training

<span id="api-madpoiterconfig"></span>
<span id="api-marlhfiterconfig"></span>

`MADPOIterTrainer` and `MARLHFIterTrainer` alternate preference collection with optimization. See [Training Framework]({{< relref "/docs/mapl/training-framework#iterative-training" >}}) for the training cycle, the [comparator]({{< relref "/docs/mapl/training-framework#comparator" >}}), and the [replay buffer]({{< relref "/docs/mapl/training-framework#replay-buffer" >}}).

## Collaboration Modes

Preference training supports separate policies per role or one actor generating the complete joint response. See [Training Framework]({{< relref "/docs/mapl/training-framework#collaboration-modes" >}}) for the supported actor/comparator layouts and centralized adapters.
