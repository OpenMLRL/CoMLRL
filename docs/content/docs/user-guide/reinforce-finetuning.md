---
title: Multi-Agent REINFORCE
weight: 2
math: true
---

## Overview

Single‑turn REINFORCE optimizes agents for one round of generation per sample. The key control is how to form joint actions from each agent's {{< katex inline=true >}}G{{< /katex >}} generations.

## Joint Mode

- align (default): pairs the g‑th generation of every agent → {{< katex inline=true >}}G{{< /katex >}} joint actions per node.
- cross: Cartesian product within a node → {{< katex inline=true >}}G^N{{< /katex >}} joint actions per node (N agents).

{{% hint %}}
Choosing align vs. cross: Align uses fewer sibling evaluations per node, leading to faster wall‑time and is a good default. Cross compares more siblings per node for better value estimation while using the same VRAM, since it reuses the same {{< katex inline=true >}}G{{< /katex >}} generations and only crosses them within the node.
{{% /hint %}}

We never cross across different nodes; this maintains causal consistency and correct credit assignment.

## MAREINFORCE

Multi‑Agent REINFORCE without a baseline.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
\Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} R^{(g)}_t \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}\mid h_{i,t})\Bigg];
{{< /katex >}}

### Variants

- MARLOO: Multi‑Agent REINFORCE Leave‑One‑Out (RLOO / Revisiting REINFORCE). Baseline is the mean return of other agents (leave‑one‑out) at the same step.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
\Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} \Big( R^{(g)}_t - \sum_{k\in \mathcal{G},\, k\neq g}\tfrac{R^{(k)}_t}{|\mathcal{G}|-1} \Big) \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}\mid h_{i,t}) \Bigg];
{{< /katex >}}

- MAReMax: Multi‑Agent REINFORCE with Group Max (ReMax). Baseline is the maximum group return at the step.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
\Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} \Big( R^{(g)}_t - \max(R_t^{\mathcal{G}}) \Big) \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}\mid h_{i,t}) \Bigg];
{{< /katex >}}

### When to use

- Simple baseline‑free training; good for small problems with dense rewards.
- Use as a reference point to compare baseline variants (MARLOO/MAReMax).

### Notes

- For sparse/noisy rewards, a baseline often stabilizes training (see variants).

### References

- RLOO (Leave‑One‑Out): https://openreview.net/forum?id=r1lgTGL5DE
- Revisiting REINFORCE: https://arxiv.org/abs/2402.14740
- ReMax: https://arxiv.org/abs/2310.10505

## MAGRPO

Multi‑Agent Group‑Relative Policy Optimization (MAGRPO) optimizes each agent with a group‑relative baseline computed among sibling joint actions at the same node/turn.

### Objective

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}\left[ \frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}}
\Big(R^{(g)}_t - \operatorname{mean}(R^{\mathcal{G}}_t)\Big)
\cdot \log \pi_{\theta_i}\big(a^{(g)}_{i,t} \mid h_{i,t}\big) \right].
{{< /katex >}}

### Siblings and baseline

- Sibling set size depends on Joint Mode and Multi‑Turn (see User Guide): align ⇒ {{< katex inline=true >}}G{{< /katex >}}, cross ⇒ {{< katex inline=true >}}G^N{{< /katex >}}.
- Group baseline is the mean over siblings at the same node/turn; this keeps the estimator unbiased and provides stable credit assignment.

### Configuration tips

- Prefer `align` initially for speed; try `cross` for more accurate estimates.
- Use modest {{< katex inline=true >}}G{{< /katex >}} (e.g., 2–4) and small `max_new_tokens` to control cost.
- Pair with a simple reward processor (e.g., scaling) to keep signals in a convenient range.

### Cost and scalability

Runtime scales with the number of siblings per node and the number of leaves. Monitor GPU memory and iteration time; reduce T, G, or token lengths as needed.

### References

- GRPO: https://arxiv.org/pdf/2402.03300
- Dr.GRPO: https://arxiv.org/abs/2503.20783
- TreeRPO: https://arxiv.org/abs/2506.05183

## Practical Tips

- Start small: {{< katex inline=true >}}G \in [2, 4]{{< /katex >}} and ensure throughput is acceptable.
- Keep prompts consistent across agents if you intend symmetric roles.
- Add light diversity to agent prompts if specialization helps the task.
- Use baseline subtraction to reduce variance in gradient estimates.

## Limitations

- Single‑turn ignores iterative refinement. For tasks benefiting from feedback cycles, use Multi‑Turn instead.
- High variance in gradient estimates may require larger batch sizes.
