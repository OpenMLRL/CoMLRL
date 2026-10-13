---
title: Overview
---

<p style="font-family: 'Futura', 'Futura PT', 'Avenir Next', 'Segoe UI', Arial, sans-serif; font-weight: 700; font-size: 1.7rem; letter-spacing: 0.em; line-height: 1; margin-top: 1.8em 0;">Welcome to CoMLRL's documentation &nbsp;👋</p>

**Co**operative **M**ulti-**L**LM **R**einforcement **L**earning (**CoMLRL**) is an open-source library for training multiple LLMs to collaborate using Multi-Agent Reinforcement Learning (MARL). It provides implementations of various MARL algorithms for LLM collaboration and supports different environments and benchmarks.

<figure class="docs-demo">
  <img src="{{< relurl "img/demo-hd.webp" >}}" width="1306" height="730" decoding="async" fetchpriority="high" alt="Animated overview of CoMLRL: pretrained language models, cooperative MARL trainers, and multi-agent environment interactions." />
</figure>

## About

{{< tabs >}}

{{% tab "Q&A" %}}

<em style="font-weight: 600; color: var(--docs-accent); margin-bottom: -0.5rem; display: block;"> "What are the differences between CoMLRL and other multi-LLM training frameworks?"</em>

A lot of works use role-conditioned, parameter-sharing approaches built on single-agent RL training framework to implement multi-agent training. Using truly distinct agents provides a better modeling where highly heterogeneous LLMs possess fundamentally different capabilities and aligns better with the concept of study — "multi-agent".

CoMLRL supports separate actors with decentralized execution, including training with a centralized critic. Its centralized collaboration mode instead uses one actor to generate the complete joint response. The available trainers include REINFORCE, Actor-Critic, and [Multi-Agent Preference Learning]({{< relref "docs/mapl" >}}); each method's page describes its implemented objective and constraints.

<em style="font-weight: 600; color: var(--docs-accent); margin-bottom: -0.5rem; display: block;"> "Does CoMLRL support single-agent fine-tuning?"</em>

<strong>Yes!</strong> The simplest way is to set `num_agents=1` in your trainer. But since we omit fancy optimizations for simplicity of multi-agent training, you may not find the single-agent trainers optimal. <a href="https://github.com/hiyouga/LLaMA-Factory">LLaMA-Factory</a>, <a href="https://github.com/huggingface/trl">trl</a>, <a href="https://github.com/OpenRLHF/OpenRLHF">OpenRLHF</a>, and <a href="https://github.com/volcengine/verl">verl</a> are good choices for single-agent fine-tuning.

<em style="font-weight: 600; color: var(--docs-accent); margin-bottom: -0.5rem; display: block;"> "Does CoMLRL support self-play/self-improvement/self-evolving by MARL?"</em>

<strong>Yes!</strong> Although we focus on LLM collaboration formalized as <a href="https://www.fransoliehoek.net/docs/OliehoekAmato16book.pdf">Dec-POMDP</a>, users can still customize the interactions with environment to implement pipeline like self-play (<a href="https://github.com/spiral-rl/spiral">Spiral</a>) and self-improvement (<a href="https://github.com/vsubramaniam851/multiagent-ft/tree/main">MAFT</a>). Users can refer to our <a href="docs/user-guide/multi-turn-training">multi-turn training</a> for more details.

<em style="font-weight: 600; color: var(--docs-accent); margin-bottom: -0.5rem; display: block;"> "Does CoMLRL support distributed training?"</em>

CoMLRL supports placing whole actor and critic models on multiple devices with `parallel_training="mp"` in one Python process. It does not provide multi-process data-parallel training or shard an individual model across devices. See [Training Parallelization]({{< relref "docs/user-guide/training-parallelization" >}}) for device configuration.

{{% /tab %}}

{{% tab "LLM Collaboration" %}}

<em style="font-weight: 600; color: var(--docs-accent);"> "What is LLM collaboration?"</em>

LLM collaboration refers to the problems where LLM agents cooperatively solve tasks in multi-agent systems. The tasks are specified in language and provided to each agent as a prompt, and the agent generates a response synchronously based on its instructions. The set of all agents' responses jointly forms a solution. Users and systems may validate the solutions to provide additional requirements or suggestions for LLMs. These components form part
of the environment for LLM collaboration, with states that may be updated based on the agents’ outputs. The updates are embedded into prompts for subsequent turns. This process iterates until the task is completed or a turn limit is reached.

{{% /tab %}}

{{% tab "MARL Fine-Tuning" %}}

<em style="font-weight: 600; color: var(--docs-accent);"> "Why should we fine-tune multi-LLM systems with MARL?"</em>

Many studies have explored LLM-based multi-agent systems for completing tasks with multiple interacting agents. However, most of these models are pretrained separately and are not specifically optimized for coordination, which would limit their performance. In addition, designing effective prompts remains difficult and unclear. Cooperative MARL methods have been extensively studied for years, which optimize a team of agents towards a shared objective. They naturally fit LLM collaboration and motivate us to bring advances from the well-established MARL community to LLM-based MAS.

{{% /tab %}}

{{% tab "Decentralization" %}}

<em style="font-weight: 600; color: var(--docs-accent);"> "What are the benefits of decentralized reasoning?"</em>

In decentralized execution, agents use their own prompts and policies, so independent generations can run concurrently when the device configuration allows it. This supports role-specific models and modular task decomposition. Latency, memory use, and task quality depend on model sizes, prompt lengths, communication, and hardware; decentralization alone does not guarantee a speedup or preserve all information available to a centralized model.

{{% /tab %}}

{{< /tabs >}}

## Features

- **Cooperative trainers for LLM collaboration:**
  - **_Multi-Agent REINFORCE_:** Critic-free policy gradient methods, including [MAREINFORCE](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/mareinforce.py), [MAGRPO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/magrpo.py), [MARLOO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/marloo.py), [MAREMAX](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/maremax.py).
    - Aligned individual response joint with `joint_mode='aligned'`.
    - Cartesian-product joint responses with `joint_mode='cross'`.
  - **_Multi-Agent Actor-Critic:_** Actor-Critic methods, including [IAC](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/actor_critic/iac.py) and [MAAC](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/actor_critic/maac.py).
    - Independent actor-critic (separate critic or value-head over LLM backbone).
    - Centralized critic over joint prompts with separate actors.
  - **_[Multi-Agent Preference Learning]({{< relref "docs/mapl" >}}):_** MADPO, MARLHF, and their iterative variants.
    - Direct joint preference optimization or a learned joint reward followed by RL.
    - Decentralized actors or one centralized actor with role-aware joint responses.
- **Environments that simulate real-world tasks for training and evaluating LLM collaboration:**
  - [**_Writing_**](https://github.com/OpenMLRL/LLM_Collab_Writing): Multiple LLM agents collaborate on processing articles.
    - [TLDR](https://huggingface.co/datasets/trl-lib/tldr) - Summarizing Reddit posts.
    - [ArXiv](http://arxiv.org/abs/1905.00075) - Expanding abstracts into introductions.
  - [**_Coding_**](https://github.com/OpenMLRL/LLM_Collab_Code_Generation): Generate code solutions for programming problems.
    - [MBPP](https://arxiv.org/abs/2108.07732) - Mostly basic python problems.
    - [HumanEval](https://arxiv.org/abs/2107.03374) - Handwritten evaluation problems.
    - [CoopHumanEval](https://huggingface.co/datasets/OpenMLRL/CoopHumanEval) - HumanEval with cooperative nature.
    - [ClassEval](https://conf.researchr.org/details/icse-2024/icse-2024-research-track/219/Evaluating-Large-Language-Models-in-Class-Level-Code-Generation) - Complete class-level code based on attributes and docstrings.
  - [**_Minecraft_**](https://github.com/OpenMLRL/LLM_Collab_Minecraft): Collaborative building tasks in Minecraft.
    - [StrBuild](https://github.com/OpenMLRL/LLM_Collab_Minecraft/tree/main/str_build) - Building structures based on string blueprints.
    - [HouseBuild](https://github.com/OpenMLRL/LLM_Collab_Minecraft/tree/main/house_build) - Constructing houses from given blueprints while defending against spider attacks.
