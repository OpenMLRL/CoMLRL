---
title: ""
---

<p style="font-family: 'Futura', 'Futura PT', 'Avenir Next', 'Segoe UI', Arial, sans-serif; font-weight: 700; font-size: 1.7rem; letter-spacing: 0.em; line-height: 1; margin-top: 1.8em 0;">Wecome to CoMLRL's documentation &nbsp;👋</p>

**Co**operative **M**ulti-**L**LM **R**einforcement **L**earning (**CoMLRL**) is a open-source library for training multiple LLMs to collaborate using Multi-Agent Reinforcement Learning (MARL). It provides implementations of various MARL algorithms for LLM collaboration and support for different environments and benchmarks.

## About

{{< tabs >}}

{{% tab "LLM Collaboration" %}}

LLM collaboration refers to the problems where LLM agents cooperatively solve tasks in multi-agent systems. The tasks are specified in language and provided to the each agent as a prompt, and the agent generates a response synchronously based on its instructions. The set of all agents' responses jointly forms a solution. Users and systems may validate the solutions to provide additional requirements or suggestions for LLMs. These components form part
of the environment for LLM collaboration, with states may updated based on the agents’ outputs. The updates are embedded into prompts for subsequent turns. This process iterates until the task is completed or a turn limit is reached.

{{% /tab %}}

{{% tab "MARL Fine-Tuning" %}}

Many studies have explored LLM-based multi-agent systems for completing tasks with multiple interacting agents. However, most of these models are pretrained separately and are not specifically optimized for coordination, which would limit their performance. In addition, designing effective prompts remains difficult and uncleared. Cooperative MARL methods have been extensively studied for years, which optimizes a team of agents towards a shared objective. They naturally fits LLM collaboration and motivates us to bring advances from well-established MARL community to LLM-based MAS.

{{% /tab %}}

{{% tab "Decentralization" %}}

Cooperative MARL methods are grounded in the theory of <a href="https://www.fransoliehoek.net/docs/OliehoekAmato16book.pdf">Dec-POMDP</a>. The agents are executing decentralizedly, which has many advantages. Unlike knowledge distillation, pruning, or quantization, it accelerates LLM inference without incurring information loss. Moreover, decentralization reduces the computational and memory burden of maintaining long-context dependencies and conducting joint decision-making within a single model. By assigning specific subtasks to individual agents, the system achieves more modular, efficient, and lightweight reasoning. In addition, effective cooperation among small local language models can offer a safe and cost-efficient solution for offline and edge intelligence.

{{% /tab %}}

{{% tab "Q&A" %}}

- <em style="font-weight: 500; color: #843b54;"> "Does CoMLRL include test-time multi-agent methods?"</em>

  No, this library primarily focuses on optimizing LLM collaboration by MARL. Designing multi-agent test-time interactions is not our strength. Users refer to <a href="https://github.com/microsoft/autogen">AutoGen</a>, <a href="https://langroid.github.io/langroid/">langroid</a>, <a href="https://github.com/TsinghuaC3I/MARTI">MARTI</a> for help.

- <em style="font-weight: 500; color: #843b54;"> "Does CoMLRL support self-play or self-improvement by MARL?"</em>

  No, we are just focusing LLM collaboration formalized as <a href="https://www.fransoliehoek.net/docs/OliehoekAmato16book.pdf">Dec-POMDP</a>. The algorithms in this library does not cover competitive or mixed-game scenarios. Users may refer to <a href="https://github.com/spiral-rl/spiral">Spiral</a> and <a href="https://github.com/vsubramaniam851/multiagent-ft/tree/main">MAFT</a> for help.

- <em style="font-weight: 500; color: #843b54;"> "Does CoMLRL support distributed training?"</em>

  No, the current propose of this library is to show the effectiveness of MARL in proof of concepts using small-scale LLMs. Distributed training for large-scale LLMs will be supported in the future.

{{% /tab %}}

{{< /tabs >}}

## Features

- **MARL trainers to optimize LLM collaboration:**
  - **_Multi-Agent REINFORCE_:** Critic-free policy gradient methods, including [MAREINFROCE](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/mareinforce.py), [MAGRPO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/magrpo.py), [MARLOO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/marloo.py), [MAREMAX](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/maremax.py).
    - Aligned individual response joint with `joint_mode='align'`.
    - Memory-efficient cross joint with `joint_mode='cross'`.
  - **_Multi-Agent PPO:_** Critic-based policy gradient methods, including [IPPO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/ippo.py).
    - Canonical IPPO with a separate critic with `use_separate_critic=True`.
    - Memory-efficient critic with value-head over actor with `use_separate_critic=False`.
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

<img src="/img/demo.gif" width="800px;" alt=""/>

# Table of Contents

## User Guide
- [Installation]({{< ref "/docs/user-guide/installation" >}})
- [Multi-Agent REINFORCE]({{< ref "/docs/user-guide/reinforce-finetuning" >}})
- [Multi-Agent PPO]({{< ref "/docs/user-guide/ppo-finetuning" >}})
- [Multi-Turn Training]({{< ref "/docs/user-guide/multi-turn" >}})

## Environments
- [Article Writing]({{< ref "/docs/env/writing-collaboration" >}})
- [Code Completion]({{< ref "/docs/env/code-completion" >}})
- [Code Generation]({{< ref "/docs/env/code-generation" >}})

## Examples
- [Quick Demo]({{< ref "/docs/examples/quick-demo" >}})

## Development
- [Contributing]({{< ref "/docs/dev/contributing" >}})
