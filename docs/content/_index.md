---
title: ""
---

<p style="font-family: 'Futura', 'Futura PT', 'Avenir Next', 'Segoe UI', Arial, sans-serif; font-weight: 700; font-size: 1.7rem; letter-spacing: 0.em; line-height: 1; margin-top: 1.8em 0;">Wecome to CoMLRL's documentation &nbsp;👋</p>

**Co**operative **M**ulti-**L**LM **R**einforcement **L**earning (**CoMLRL**) is a open-source library for training multiple LLMs to collaborate using Multi-Agent Reinforcement Learning (MARL). It provides implementations of various MARL algorithms for LLM collaboration and support for different environments and benchmarks.

{{< tabs >}}

{{% tab "LLM Collaboration" %}}

LLM collaboration refers to the problems where LLM agents cooperatively solve tasks in multi-agent systems. The tasks are specified in natural language and provided to the each agent as a prompt, and the agent generates a response synchronously based on its individual instructions. The set of these responses jointly forms a solution. Most tasks cannot be resolved in one turn. Users and systems validate the solutions to provide additional requirements or suggestions for LLMs. These components also serve as part
of the environment for LLM collaboration, whose states may change based on the agents’ outputs. The updates are embedded into prompts for subsequent turns. This iterative process continues until the task is completed or a turn limit is reached.

{{% /tab %}}

{{% tab "MARL Fine-Tuning" %}}

{{% /tab %}}

{{% tab "Decentralization" %}}

{{% /tab %}}

{{% tab "Q&A" %}}

{{% /tab %}}

{{< /tabs >}}

## Features

- We develop many trainers.
- Examples of LLM collaboration is in the scenarios.

<img src="/img/demo.gif" width="800px;" alt=""/>
