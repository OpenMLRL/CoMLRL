---
title: Multi-Agent REINFORCE
weight: 2
math: true
---

REINFORCE optimizes the policy directly using sampled returns. An action-independent baseline can be involved to reduce variance for REINFORCE methods. REINFORCE methods have been widely used to fine-tune LLMs, because of their simplicity and effectiveness, e.g, [GRPO](https://arxiv.org/pdf/2402.03300),[Dr. GRPO](https://arxiv.org/abs/2503.20783), [RLOO](https://openreview.net/forum?id=r1lgTGL5DE), [ReMax](https://arxiv.org/abs/2310.1050), [TreeRPO](https://arxiv.org/abs/2506.05183), and [REINFROCE++](https://arxiv.org/abs/2501.03262).

## MAREINFORCE

In the LLM collaboration setting, REINFORCE can be extended to optimize each agent's policy with joint returns from multiple agents.

- **MAREINFORCE**: The naive Multi‑Agent REINFORCE without a baseline can be expressed by:

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
\Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} R^{(g)}_t \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}\mid h_{i,t})\Bigg];
{{< /katex >}}

This class is derived from `comlrl.trainers.magrpo.MAGRPOTrainer`. Interfaces for the trainer and configuration classes are same as `MAGRPOTrainer` and `MAGRPOConfig`.

## MAGRPO

Multi‑Agent Group‑Relative Policy Optimization optimizes each agent with a group‑relative baseline computed among sibling joint actions at the same node.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}\left[ \frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}}
\Big(R^{(g)}_t - \operatorname{mean}(R^{\mathcal{G}}_t)\Big)
\cdot \log \pi_{\theta_i}\big(a^{(g)}_{i,t} \mid h_{i,t}\big) \right].
{{< /katex >}}



## Other Variants

CoMLRL also implements other Multi-Agent REINFORCE variants with different baselines:

- **MARLOO**: Multi‑Agent REINFORCE Leave‑One‑Out. Baseline is the mean return of other agents (leave‑one‑out) at the same step.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
\Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} \Big( R^{(g)}_t - \sum_{k\in \mathcal{G},\, k\neq g}\tfrac{R^{(k)}_t}{|\mathcal{G}|-1} \Big) \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}\mid h_{i,t}) \Bigg];
{{< /katex >}}

- **MAReMax**: Multi‑Agent REINFORCE with Group Max. Baseline is the maximum group return at the step.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
\Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} \Big( R^{(g)}_t - \max(R_t^{\mathcal{G}}) \Big) \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}\mid h_{i,t}) \Bigg];
{{< /katex >}}

These classes are derived from `comlrl.trainers.magrpo.MAGRPOTrainer`. Interfaces for the trainer and configuration classes are same as `MAGRPOTrainer` and `MAGRPOConfig`.
