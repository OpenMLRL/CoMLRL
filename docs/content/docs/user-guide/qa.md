---
title: "Q&A"
layout: qa
bookToC: false
weight: 7
aliases:
  - /docs/qa/
---

{{% details title="What are the differences between CoMLRL and other multi-LLM training frameworks?" id="what-are-the-differences-between-comlrl-and-other-multi-llm-training-frameworks" class="docs-qa-item" %}}

Many approaches use single-agent RL frameworks to train role-conditioned agents that share parameters. Distinct agents better represent heterogeneous LLMs with different capabilities and reflect the multi-agent setting more directly.

CoMLRL supports separate actors with decentralized execution, including training with a centralized critic. Its centralized collaboration mode instead uses one actor to generate the complete joint response. The available trainers include REINFORCE, Actor-Critic, and Multi-Agent PbRL; each method's page describes its implemented objective and constraints.

{{% /details %}}

{{% details title="Does CoMLRL support single-agent fine-tuning?" id="does-comlrl-support-single-agent-fine-tuning" class="docs-qa-item" %}}

<strong>Yes.</strong> Users can set `num_agents=1` for single-agent fine-tuning. CoMLRL prioritizes simple multi-agent training and omits some optimizations, so its single-agent trainers may not be optimal. <a href="https://github.com/hiyouga/LLaMA-Factory">LLaMA-Factory</a>, <a href="https://github.com/huggingface/trl">trl</a>, <a href="https://github.com/OpenRLHF/OpenRLHF">OpenRLHF</a>, and <a href="https://github.com/volcengine/verl">verl</a> are alternatives for single-agent fine-tuning.

{{% /details %}}

{{% details title="Does CoMLRL support self-play/self-improvement/self-evolution with MARL?" id="does-comlrl-support-self-playself-improvementself-evolution-with-marl" class="docs-qa-item" %}}

<strong>Yes.</strong> Although CoMLRL focuses on LLM collaboration formalized as a <a href="https://www.fransoliehoek.net/docs/OliehoekAmato16book.pdf">Dec-POMDP</a>, users can customize environment interactions to implement pipelines such as self-play (<a href="https://github.com/spiral-rl/spiral">Spiral</a>) and self-improvement (<a href="https://github.com/vsubramaniam851/multiagent-ft/tree/main">MAFT</a>). The <a href="{{< relref "docs/multi-turn/algorithm" >}}">multi-turn training</a> documentation provides further details.

{{% /details %}}

{{% details title="Does CoMLRL support distributed training?" id="does-comlrl-support-distributed-training" class="docs-qa-item" %}}

CoMLRL can place whole models, such as actors, critics, and reward models, on different GPUs with `parallel_training="mp"` in one Python process. It does not provide multi-process data-parallel training or shard a single model across devices. [Training Parallelization]({{< relref "docs/user-guide/training-parallelization" >}}) describes device configuration.

{{% /details %}}
