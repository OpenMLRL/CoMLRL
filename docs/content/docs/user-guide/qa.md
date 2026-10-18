---
title: "Q&A"
layout: qa
bookToC: false
weight: 7
aliases:
  - /docs/qa/
---

{{% details title="What are the differences between CoMLRL and other multi-LLM training frameworks?" id="what-are-the-differences-between-comlrl-and-other-multi-llm-training-frameworks" class="docs-qa-item" %}}

Many frameworks support training role-specific LLM-based multi-agent systems ([PettingLLMs](https://github.com/pettingllms-ai/PettingLLMs), [Dr. MAS](https://github.com/langfengQ/DrMAS), [MARTI](https://github.com/TsinghuaC3I/MARTI)). Most of them are formulated as partially observable stochastic games (POSGs), in which each agent has its own reward. Hand-crafted reward rules for each agent or role make training signals denser, but they are onerous to design.

[Cooperative MARL](https://arxiv.org/abs/2405.06161) has been studied for a long time and is normally formulated as a [Dec-POMDP](https://www.fransoliehoek.net/docs/OliehoekAmato16book.pdf). In a Dec-POMDP, all agents share one reward, so the optimal joint policy is, in theory, globally optimal for the collaboration. This formulation also allows agents to find more specific tasks from loosely divided roles (see the [CoopHumanEval](https://huggingface.co/datasets/OpenMLRL/CoopHumanEval) coding domain). In addition, unlike parameter-sharing approaches, CoMLRL mainly supports optimizing systems of truly distinct LLM agents, which better represent heterogeneous LLMs with different capabilities and directly reflect the multi-agent setting.

{{% /details %}}

{{% details title="Does CoMLRL support single-agent RL post-training?" id="does-comlrl-support-single-agent-rl-post-training" class="docs-qa-item" %}}

<strong>Yes.</strong> Users can set `num_agents=1` for single-agent RL post-training by following the <a href="{{< relref "docs/user-guide/training-initialization" >}}">Training Initialization</a> guide. However, CoMLRL prioritizes simple multi-agent training and omits some optimizations specific to the single-agent case. For single-agent training, <a href="https://github.com/hiyouga/LLaMA-Factory">LLaMA-Factory</a>, <a href="https://github.com/huggingface/trl">trl</a>, <a href="https://github.com/OpenRLHF/OpenRLHF">OpenRLHF</a>, and <a href="https://github.com/volcengine/verl">verl</a> are better choices.

{{% /details %}}

{{% details title="Does CoMLRL support self-play or RSI with RL post-training?" id="does-comlrl-support-self-play-or-rsi-with-rl-post-training" class="docs-qa-item" %}}

<strong>Yes.</strong> CoMLRL supports <a href="{{< relref "docs/marl/multi-turn-interaction" >}}">multi-turn RL post-training</a>, so environment interactions can be customized for these pipelines.

Self-play (<a href="https://github.com/spiral-rl/spiral">Spiral</a>, <a href="https://github.com/spade-rl/spade">SPADE</a>) and self-improvement (<a href="https://github.com/vsubramaniam851/multiagent-ft/tree/main">MAFT</a>) can be implemented with a single agent in one trainer (`num_agents=1`), whose external transition adjusts a self-play-style prompt at each turn, for example by switching the agent's role or asking it to revise its previous answer.

Multi-agent recursive self-improvement (RSI) is more involved: each agent runs in its own trainer, and the system coordinates these trainers. At each turn, the system sends different prompts to one or more trainers to collect their agents' responses, and it triggers each trainer's training at its scheduled steps, for example by alternating between agents.

{{% /details %}}

{{% details title="Does CoMLRL support distributed training?" id="does-comlrl-support-distributed-training" class="docs-qa-item" %}}

CoMLRL can place whole models, such as actors, critics, and reward models, on different GPUs of a single machine with `parallel_training="mp"`. It does not provide multi-process data-parallel training or shard a single model across devices. [Training Parallelization]({{< relref "docs/user-guide/training-parallelization" >}}) describes device configuration.

{{% /details %}}

{{% details title="Can I use parameter sharing to save VRAM?" id="can-i-use-parameter-sharing-to-save-vram" class="docs-qa-item" %}}

<strong>Yes.</strong> As described in [Training Entities]({{< relref "docs/user-guide/training-entities" >}}), `agents` accepts loaded model objects, so one loaded model can be passed for every agent. The agents then share its weights, and formatters can give them different roles. Each agent still keeps its own optimizer state, so sharing saves memory for weights and gradients but not for optimizer states.

{{% /details %}}

{{% details title="Can I use LoRA or lower-precision training to save VRAM?" id="can-i-use-lora-or-lower-precision-training-to-save-vram" class="docs-qa-item" %}}

<strong>Not recommended.</strong> In our extensive experiments, neither PEFT methods such as LoRA nor lower-precision training worked well. LoRA could not fully tune the models, and effective changes may even require larger LoRA adapters, which can end up increasing memory use. Instead, parameter sharing can be used when agents start from models with similar initial performance, and gradient checkpointing can reduce activation memory, although these approaches have not been widely tested. Users are welcome to test them and share their findings in the [CoMLRL issues](https://github.com/OpenMLRL/CoMLRL/issues).

{{% /details %}}
