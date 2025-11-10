---
title: Multi-Turn Training
linkTitle: Multi-Turn Training
weight: 4
math: true
---

## Overview

Multi‑turn training allows agents to iteratively refine outputs over {{< katex inline=true >}}T{{< /katex >}} turns. The number of turns ({{< katex inline=true >}}T{{< /katex >}}) and generations per turn ({{< katex inline=true >}}G{{< /katex >}}) determine search breadth and cost.

## Branching and complexity

- Align: total leaves ~ {{< katex inline=true >}}G^T{{< /katex >}}
- Cross: total leaves ~ {{< katex inline=true >}}G^{N \cdot T}{{< /katex >}} (N agents)

At each node (turn), the sibling set size is {{< katex inline=true >}}G{{< /katex >}} for align and {{< katex inline=true >}}G^N{{< /katex >}} for cross. Increase {{< katex inline=true >}}T{{< /katex >}} and/or {{< katex inline=true >}}G{{< /katex >}} to explore more paths, trading off wall‑time and evaluation cost.

## Joint Mode (recap)

- align (default): pairs the g‑th generation of every agent → {{< katex inline=true >}}G{{< /katex >}} joint actions per node.
- cross: Cartesian product within a node → {{< katex inline=true >}}G^N{{< /katex >}} joint actions per node (N agents).

## External feedback

Controls how environment feedback is attached to the next turn’s prompts.

- Mode (`external.mode`): default `level_feedback` (adds diagnostics)
  - expert_edits: an LLM proposes edits; prompts include edit suggestions + context
  - plain: no diagnostics; still includes previous response and a “revise” instruction

- Sandbox slice (`external.sandbox_slice`): how many eval tests to include in feedback (analysis‑based modes)
  - 1 (default): first assert only
  - 0, None, or 'all': include all eval tests
  - Negative values: use the last asserts
  - No effect in `expert_edits` mode

- Expert edits model (`external.expert_edits_model`): model used to propose edits in `expert_edits` mode (default: DeepSeek‑Coder). Can be changed to Claude, GPT, etc., when keys/tokens are configured.

## Termination

Early‑stop criterion per node/turn based on mean immediate reward over sibling joint actions. If the mean exceeds the threshold, the branch stops expanding at this turn and backpropagates from the truncated subtree. Other branches continue.

## Practical settings

- Start with {{< katex inline=true >}}T = 2{{< /katex >}} and {{< katex inline=true >}}G \in [2, 4]{{< /katex >}} to validate the pipeline.
- Use `align` for faster iterations; try `cross` when estimation quality matters.
- If feedback is noisy, reduce `external.sandbox_slice` or simplify prompts.

## Common pitfalls

- Exploding cost with high {{< katex inline=true >}}T{{< /katex >}}/{{< katex inline=true >}}G{{< /katex >}}: cap `max_new_tokens` and reduce generations.
- Misaligned prompts across agents lead to unstable credit assignment.
