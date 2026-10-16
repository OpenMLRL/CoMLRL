---
title: Environments
weight: 2
bookCollapseSection: false
bookFlatSection: true
---

Environments that simulate real-world tasks for training and evaluating LLM collaboration:

- **Writing**: Multiple LLM agents collaborate on processing articles.
  - [TLDR](https://huggingface.co/datasets/trl-lib/tldr) - Summarizing Reddit posts.
  - [arXiv](https://arxiv.org/abs/1905.00075) - Expanding abstracts into introductions.

- **Coding**: Collaborative code generation for programming problems.
  - [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) - Mostly basic Python problems.
  - [HumanEval](https://github.com/openai/human-eval) - Handwritten evaluation problems.
  - [CoopHumanEval](https://huggingface.co/datasets/OpenMLRL/CoopHumanEval) - HumanEval with a cooperative nature.

- **Minecraft**: Multi-agent building environments.
  - StrBuild - Structured builds from text.
  - HouseBuild - Coordinated house construction.

- **Travel**: Collaborative travel planning under user constraints.
  - [TravelPlanner](https://huggingface.co/datasets/osunlp/TravelPlanner) - Planning multi-day trip itineraries.

- **Tool-Use**: Collaborative function calling with tools using [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html).
  - Native Parallel - Making parallel function calls.
