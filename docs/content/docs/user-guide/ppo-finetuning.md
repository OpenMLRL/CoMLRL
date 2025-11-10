---
title: Multi-Agent PPO
weight: 3
math: true
---

## Overview

Single‑turn PPO (Proximal Policy Optimization) provides more stable training compared to REINFORCE by limiting policy updates. It optimizes agents for one round of generation per sample.

## Joint Mode

- align (default): pairs the g‑th generation of every agent → {{< katex inline=true >}}G{{< /katex >}} joint actions per node.
- cross: Cartesian product within a node → {{< katex inline=true >}}G^N{{< /katex >}} joint actions per node (N agents).

{{% hint %}}
Choosing align vs. cross: Align uses fewer sibling evaluations per node, leading to faster wall‑time and is a good default. Cross compares more siblings per node for better value estimation while using the same VRAM, since it reuses the same {{< katex inline=true >}}G{{< /katex >}} generations and only crosses them within the node.
{{% /hint %}}

We never cross across different nodes; this maintains causal consistency and correct credit assignment.

## Algorithm Details

PPO uses a clipped objective function to prevent large policy updates, resulting in more stable training. The algorithm alternates between sampling data and performing multiple epochs of optimization on the sampled data.

## Practical Tips

- Start small: {{< katex inline=true >}}G \in [2, 4]{{< /katex >}} and ensure throughput is acceptable.
- Keep prompts consistent across agents if you intend symmetric roles.
- Add light diversity to agent prompts if specialization helps the task.
- Tune the clipping parameter (typically 0.1-0.3) for optimal stability vs. learning speed.
- Multiple optimization epochs can improve sample efficiency.

## Limitations

- Single‑turn ignores iterative refinement. For tasks benefiting from feedback cycles, use Multi‑Turn instead.
- Requires more memory due to the need to store old policy probabilities.
