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

Cooperative MARL methods are grounded in the theory of Dec-POMDP. The agents are executing decentralizedly, which has many advantages. First, unlike knowledge distillation, pruning, or quantization, it accelerates LLM inference without incurring information loss. Moreover, decentralization reduces the computational and memory burden of maintaining long-context dependencies and conducting joint decision-making within a single model. By assigning specific subtasks to individual agents, the system achieves more modular, efficient, and lightweight reasoning. In addition, effective cooperation among small local language models can offer a safe and cost-efficient solution for offline and edge intelligence.

{{% /tab %}}

{{% tab "Q&A" %}}

- <em style="font-weight: 500; color: #a45b74;"> "Do you have multi-agent test-time methods?"</em>

  This project primarily focuses on optimizing LLM collaboration by MARL fine-tuning, multi-agent test-time cooperation is not our strength. We recommend users refer to AutoGen, LangChain, MARTI.

- <em style="font-weight: 500; color: #a45b74;"> "Do you have multi-agent self-play?"</em>

  This project does not cover competitive or mixed-game scenarios.

{{% /tab %}}

{{< /tabs >}}

## Features

- We support various classical MARL trainers.
- Examples of LLM collaboration is in the scenarios.

<img src="/img/demo.gif" width="800px;" alt=""/>
