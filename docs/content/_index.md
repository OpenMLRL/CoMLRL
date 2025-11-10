---
title: ""
---

<p style="font-family: 'Futura', 'Futura PT', 'Avenir Next', 'Segoe UI', Arial, sans-serif; font-weight: 700; font-size: 1.7rem; letter-spacing: 0.em; line-height: 1; margin-top: 1.8em 0;">Wecome to CoMLRL's documentation &nbsp;👋</p>

**Co**operative **M**ulti-**L**LM **R**einforcement **L**earning (**CoMLRL**) is a open-source library for training multiple LLMs to collaborate using Multi-Agent Reinforcement Learning (MARL). It provides implementations of various MARL algorithms for LLM collaboration and support for different environments and benchmarks.

{{< tabs >}}

{{% tab "LLM Collaboration" %}}

LLM collaboration refers to the problems where LLM agents cooperatively solve tasks in multi-agent systems. The tasks are specified in language and provided to the each agent as a prompt, and the agent generates a response synchronously based on its individual instructions. The set of all these responses jointly forms a solution. Most tasks cannot be resolved in one turn. Users and systems validate the solutions to provide additional requirements or suggestions for LLMs. These components serve as part
of the environment for LLM collaboration, whose states may change based on the agents’ outputs. The updates are embedded into prompts for subsequent turns. This iterative process continues until the task is completed or a turn limit is reached.

{{% /tab %}}

{{% tab "MARL Fine-Tuning" %}}

Many studies have explored LLM-based multi-agent systems for completing tasks with multiple interacting agents. However, most of these models are pretrained separately and are not specifically optimized for coordination, which limits their performance. In addition, designing effective prompts remains difficult and uncleared. Cooperative MARL methods have been extensively studied, which optimizes a team of agents towards a shared objective. They naturally fits LLM collaboration and motivates us to bring advances from well-established MARL community to LLM-based multi-agent systems.

{{% /tab %}}

{{% tab "Decentralization" %}}

{{% /tab %}}

{{% tab "Q&A" %}}

<em style="font-weight: 500; color: #a45b74;"> "Do you have multi-agent inference-time interaction?"</em>

This project focuses on LLM collaboration and does not cover competitive or mixed-game scenarios.

{{% /tab %}}

{{< /tabs >}}

## Features

- We develop many trainers.
- Examples of LLM collaboration is in the scenarios.

<img src="/img/demo.gif" width="800px;" alt=""/>
