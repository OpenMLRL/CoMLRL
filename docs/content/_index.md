---
title: Overview
---

<h1 class="docs-welcome">Welcome to CoMLRL's documentation</h1>

**Co**operative **M**ulti-**L**LM **R**einforcement **L**earning (**CoMLRL**) is an open-source library for training multiple LLMs to collaborate using Multi-Agent Reinforcement Learning (MARL). It implements MARL algorithms for LLM collaboration and supports a variety of environments and benchmarks.

<figure class="docs-demo">
  <img src="{{< relurl "img/demo-hd.webp" >}}" width="1306" height="730" decoding="async" fetchpriority="high" alt="Animated overview of CoMLRL: pretrained language models, cooperative MARL trainers, and multi-agent environment interactions." />
</figure>

## About

{{< tabs >}}

{{% tab "LLM Collaboration" %}}

<em style="font-weight: 600; color: var(--docs-accent);"> "What is LLM collaboration?"</em>

LLM collaboration involves agents working together to solve tasks in multi-agent systems. Each agent receives a task prompt and responds synchronously according to its instructions, and the agents' responses jointly form a solution. Users or external systems may then validate the solution and add requirements or suggestions. These interactions are part of the environment, whose state can change with the agents' outputs; the updated state enters the prompts of later turns until the task is complete or a turn limit is reached.

{{% /tab %}}

{{% tab "MARL Fine-Tuning" %}}

<em style="font-weight: 600; color: var(--docs-accent);"> "Why should we fine-tune multi-LLM systems with MARL?"</em>

Many studies solve tasks with LLM-based multi-agent systems through agent interaction. However, most models are pretrained separately and never optimized for coordination, which can limit performance, and effective prompts remain hard to design. Cooperative MARL optimizes a team of agents toward a shared objective, which makes it a natural foundation for training LLM-based multi-agent systems.

{{% /tab %}}

{{% tab "Decentralization" %}}

<em style="font-weight: 600; color: var(--docs-accent);"> "What are the benefits of decentralized reasoning?"</em>

During decentralized execution, each agent uses its own prompt and policy. Independent generations can run concurrently when the devices permit, which supports role-specific models and modular task decomposition. Latency, memory use, and task quality still depend on model sizes, prompt lengths, communication, and devices; decentralization alone does not guarantee a speedup, nor does it give each agent all the information available to a centralized model.

{{% /tab %}}

{{< /tabs >}}

## Features

- **Multi-agent learning methods for LLM collaboration:**
  - **_Multi-Agent REINFORCE:_** Critic-free policy gradient methods, including [MAREINFORCE](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/mareinforce.py), [MAGRPO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/magrpo.py), [MARLOO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/marloo.py), and [MAReMax](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/maremax.py).
    - Index-aligned joint responses with `joint_mode='aligned'`.
    - Cartesian-product joint responses with `joint_mode='cross'`.
  - **_Multi-Agent Actor-Critic:_** Actor-Critic methods, including [IAC](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/actor_critic/iac.py) and [MAAC](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/actor_critic/maac.py).
    - Independent actor-critic (separate critic or value head over the LLM backbone).
    - Centralized critic over joint prompts with separate actors.
  - **_Multi-Agent PbRL:_** MADPO, MARLHF, and their iterative variants.
    - Direct joint preference optimization or a learned joint reward followed by RL.
    - Decentralized actors or one centralized actor with role-aware joint responses.
- **Environments that simulate real-world tasks for training and evaluating LLM collaboration:**
  - [**_Writing_**](https://github.com/OpenMLRL/LLM_Collab_Writing): Multiple LLM agents collaborate to process articles.
    - [TLDR](https://huggingface.co/datasets/trl-lib/tldr) - Summarizing Reddit posts.
    - [arXiv](https://arxiv.org/abs/1905.00075) - Expanding abstracts into introductions.
  - [**_Coding_**](https://github.com/OpenMLRL/LLM_Collab_Code_Generation): Collaborative code generation for programming problems.
    - [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) - Mostly basic Python problems.
    - [HumanEval](https://github.com/openai/human-eval) - Handwritten evaluation problems.
    - [CoopHumanEval](https://huggingface.co/datasets/OpenMLRL/CoopHumanEval) - A cooperative variant of HumanEval.
    - [ClassEval](https://conf.researchr.org/details/icse-2024/icse-2024-research-track/219/Evaluating-Large-Language-Models-in-Class-Level-Code-Generation) - Completing class-level code based on attributes and docstrings.
  - [**_Minecraft_**](https://github.com/OpenMLRL/LLM_Collab_Minecraft): Collaborative building tasks in Minecraft.
    - [StrBuild](https://github.com/OpenMLRL/LLM_Collab_Minecraft/tree/main/str_build) - Building structures based on string blueprints.
    - [HouseBuild](https://github.com/OpenMLRL/LLM_Collab_Minecraft/tree/main/house_build) - Constructing houses from given blueprints while defending against spider attacks.
  - [**_Travel_**](https://github.com/OpenMLRL/LLM_Collab_Travel): Collaborative travel planning under user constraints.
    - [TravelPlanner](https://huggingface.co/datasets/osunlp/TravelPlanner) - Planning multi-day trip itineraries.
  - [**_Tool-Use_**](https://github.com/OpenMLRL/LLM_Collab_BFCL-v4): Collaborative function calling on [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html).
    - [Native Parallel](https://huggingface.co/datasets/OpenMLRL/BFCL-V4-Parallel-Native) - Making parallel function calls.
