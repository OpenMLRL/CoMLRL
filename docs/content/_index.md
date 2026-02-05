---
title: ""
---

<p style="font-family: 'Futura', 'Futura PT', 'Avenir Next', 'Segoe UI', Arial, sans-serif; font-weight: 700; font-size: 1.7rem; letter-spacing: 0.em; line-height: 1; margin-top: 1.8em 0;">Welcome to CoMLRL's documentation &nbsp;👋</p>

**Co**operative **M**ulti-**L**LM **R**einforcement **L**earning (**CoMLRL**) is an open-source library for training multiple LLMs to collaborate using Multi-Agent Reinforcement Learning (MARL). It provides implementations of various MARL algorithms for LLM collaboration and supports different environments and benchmarks.

## About

{{< tabs >}}

{{% tab "Q&A" %}}

<em style="font-weight: 600; color: #9555af; margin-bottom: -0.5rem; display: block;"> "What are the differences between CoMLRL and other multi-LLM training frameworks?"</em>

A lot of works use role-conditioned, parameter-sharing approaches built on single-agent RL training framework to implement multi-agent training. Using truly distinct agents provides a better modeling where highly heterogeneous LLMs possess fundamentally different capabilities and aligns better with the concept of study — "multi-agent".

Compare with other multi-LLM training frameworks, agents can either be trained centralizedly or decentralizedly with CoMLRL, while their execution is always fully decentralized to ensure efficient inference. CoMLRL implements standard MARL algorithms from scratch to maximize flexibility and customizability while maintaining simplicity for usage.

<em style="font-weight: 600; color: #9555af; margin-bottom: -0.5rem; display: block;"> "Does CoMLRL support single-agent fine-tuning?"</em>

<strong>Yes!</strong> The simplest way is to set `num_agents=1` in your trainer. But since we omit fancy optimizations for simplicity of multi-agent training, you may not find the single-agent trainers optimal. <a href="https://github.com/hiyouga/LLaMA-Factory">LLaMA-Factory</a>, <a href="https://github.com/huggingface/trl">trl</a>, <a href="https://github.com/OpenRLHF/OpenRLHF">OpenRLHF</a>, and <a href="https://github.com/volcengine/verl">verl</a> are good choices for single-agent fine-tuning.

<em style="font-weight: 600; color: #9555af; margin-bottom: -0.5rem; display: block;"> "Does CoMLRL support self-play/self-improvement/self-evolving by MARL?"</em>

<strong>Yes!</strong> Although we focus on LLM collaboration formalized as <a href="https://www.fransoliehoek.net/docs/OliehoekAmato16book.pdf">Dec-POMDP</a>, users can still customize the interactions with environment to implement pipeline like self-play (<a href="https://github.com/spiral-rl/spiral">Spiral</a>) and self-improvement (<a href="https://github.com/vsubramaniam851/multiagent-ft/tree/main">MAFT</a>). Users can refer to our <a href="docs/user-guide/multi-turn">multi-turn training</a> for more details.

<em style="font-weight: 600; color: #9555af; margin-bottom: -0.5rem; display: block;"> "Does CoMLRL support distributed training?"</em>

<strong>Not yet.</strong> We are currently focusing on <a href="https://arxiv.org/abs/2409.03052">CTDE</a> on light-weighted training small-scale LLMs with cooperative MARL for proof of the concept. Resource-consuming distributed training with slow and complex gradient accumulation is under development and will be open-sourced in the near future.

{{% /tab %}}

{{% tab "LLM Collaboration" %}}

<em style="font-weight: 600; color: #9555af;"> "What is LLM collaboration?"</em>

LLM collaboration refers to the problems where LLM agents cooperatively solve tasks in multi-agent systems. The tasks are specified in language and provided to each agent as a prompt, and the agent generates a response synchronously based on its instructions. The set of all agents' responses jointly forms a solution. Users and systems may validate the solutions to provide additional requirements or suggestions for LLMs. These components form part
of the environment for LLM collaboration, with states that may be updated based on the agents’ outputs. The updates are embedded into prompts for subsequent turns. This process iterates until the task is completed or a turn limit is reached.

{{% /tab %}}

{{% tab "MARL Fine-Tuning" %}}

<em style="font-weight: 600; color: #9555af;"> "Why should we fine-tune multi-LLM systems with MARL?"</em>

Many studies have explored LLM-based multi-agent systems for completing tasks with multiple interacting agents. However, most of these models are pretrained separately and are not specifically optimized for coordination, which would limit their performance. In addition, designing effective prompts remains difficult and unclear. Cooperative MARL methods have been extensively studied for years, which optimize a team of agents towards a shared objective. They naturally fit LLM collaboration and motivate us to bring advances from the well-established MARL community to LLM-based MAS.

{{% /tab %}}

{{% tab "Decentralization" %}}

<em style="font-weight: 600; color: #9555af;"> "What are the benefits of decentralized reasoning?"</em>

Cooperative MARL methods are grounded in the theory of <a href="https://www.fransoliehoek.net/docs/OliehoekAmato16book.pdf">Dec-POMDP</a>. The agents execute in a decentralized manner, which has many advantages. Unlike knowledge distillation, pruning, or quantization, it accelerates LLM inference without incurring information loss. Moreover, decentralization reduces the computational and memory burden of maintaining long-context dependencies and conducting joint decision-making within a single model. By assigning specific subtasks to individual agents, the system achieves more modular, efficient, and lightweight reasoning. In addition, effective cooperation among small local language models can offer a safe and cost-efficient solution for offline and edge intelligence.

{{% /tab %}}

{{< /tabs >}}

## Features

- **MARL trainers to optimize LLM collaboration:**
  - **_Multi-Agent REINFORCE_:** Critic-free policy gradient methods, including [MAREINFORCE](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/mareinforce.py), [MAGRPO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/magrpo.py), [MARLOO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/marloo.py), [MAREMAX](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/reinforce/maremax.py).
    - Aligned individual response joint with `joint_mode='align'`.
    - Memory-efficient cross joint with `joint_mode='cross'`.
  - **_Multi-Agent Actor-Critic:_** Critic-based policy gradient methods, including [IAC](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/actor_critic/iac.py) and [MAAC](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/actor_critic/maac.py).
    - Independent actor-critic (separate critic or value-head over LLM backbone).
    - Centralized critic over joint prompts with separate actors.
- **Environments that simulate real-world tasks for training and evaluating LLM collaboration:**
  - [**_Writing Collaboration_**](https://github.com/OpenMLRL/LLM_Collab_Writing): Multiple LLM agents collaborate on processing articles.
    - [TLDR](https://huggingface.co/datasets/trl-lib/tldr) - Summarizing Reddit posts.
    - [ArXiv](http://arxiv.org/abs/1905.00075) - Expanding abstracts into introductions.
  - [**_Code Generation_**](https://github.com/OpenMLRL/LLM_Collab_Code_Generation): Generate code solutions for programming problems.
    - [MBPP](https://arxiv.org/abs/2108.07732) - Mostly basic python problems.
    - [HumanEval](https://arxiv.org/abs/2107.03374) - Handwritten evaluation problems
    - [CoopHumanEval](https://huggingface.co/datasets/OpenMLRL/CoopHumanEval) - HumanEval with cooperative nature.
  - [**_Code Completion_**](https://github.com/OpenMLRL/LLM_Collab_Code_Completion): Complete code snippets based on given contexts.
    - [ClassEval](https://conf.researchr.org/details/icse-2024/icse-2024-research-track/219/Evaluating-Large-Language-Models-in-Class-Level-Code-Generation) - Complete class-level code based on method stubs and docstrings.


<p align="center">
  <img src="/img/demo.gif" width="800px;" alt=""/>
</p>
