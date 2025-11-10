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

- <em style="font-weight: 500; color: #a45b74;"> "Do you have test-time multi-agent methods?"</em>

  No, this library primarily focuses on optimizing LLM collaboration by MARL. Designing multi-agent test-time interactions is not our strength. Users refer to <a href="https://github.com/microsoft/autogen">AutoGen</a>, <a href="https://langroid.github.io/langroid/">langroid</a>, <a href="https://github.com/TsinghuaC3I/MARTI">MARTI</a> for help.

- <em style="font-weight: 500; color: #a45b74;"> "Do you support self-play or self-improvement by MARL?"</em>

  No, we are just focusing LLM collaboration formalized as <a href="https://www.fransoliehoek.net/docs/OliehoekAmato16book.pdf">Dec-POMDP</a>. The algorithms in this library does not cover competitive or mixed-game scenarios. Users may refer to <a href="https://github.com/spiral-rl/spiral">Spiral</a> and <a href="https://github.com/vsubramaniam851/multiagent-ft/tree/main">Multi-Agent Fine-Tuning</a> for help.

- <em style="font-weight: 500; color: #a45b74;"> "Do you support distributed training?"</em>

  No, the current propose of this library is to show the effectiveness of MARL in proof of concepts using small-scale LLMs. Distributed training for large-scale LLMs will be supported in the future.

{{% /tab %}}

{{< /tabs >}}

## Features

- We support various classical MARL trainers.
- Examples of LLM collaboration is in the scenarios.

<img src="/img/demo.gif" width="800px;" alt=""/>
