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
\Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} R^{(g)}_t \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}\mid h_{i,t})\Bigg].
{{< /katex >}}

{{% hint success %}}
These classes are derived from `comlrl.trainers.magrpo.MAGRPOTrainer`. Interfaces for the trainer and configuration classes are same as `MAGRPOTrainer` and `MAGRPOConfig`.
{{% /hint %}}

## MAGRPO

Multi‑Agent Group‑Relative Policy Optimization optimizes each agent with a group‑relative baseline computed among sibling joint actions at the same node.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}\left[ \frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}}
\Big(R^{(g)}_t - \operatorname{mean}(R^{\mathcal{G}}_t)\Big)
\cdot \log \pi_{\theta_i}\big(a^{(g)}_{i,t} \mid h_{i,t}\big) \right].
{{< /katex >}}

{{% hint info %}}
**MAGRPOConfig** inherits from `TrainingArguments` and provides parameters for both single-turn and multi-turn training:

- `num_agents`: Number of agents (default: 2)
- `num_generations`: Number of generations to sample per prompt for each agent (default: 4)
- `max_new_tokens`: Maximum number of new tokens to generate (default: 256)
- `temperature`: Temperature for sampling (default: 0.7)
- `top_p`: Top-p for sampling (default: 0.9)
- `num_turns`: Number of turns per episode; set >1 for multi-turn (default: 1)
- `discount`: Discount factor gamma over turns for returns (default: 0.9)
- `joint_mode`: Joint action composition - `'aligned'` (index-aligned, default) or `'cross'` (Cartesian product)
- `termination_threshold`: Early stop a branch if mean reward exceeds this threshold (default: None)
- `eval_interval`: Run evaluation every N training batches (default: 4)
- `eval_num_samples`: Number of samples to evaluate per evaluation run (default: 4)
{{% /hint %}}

{{% hint info %}}
**MAGRPOTrainer** accepts either a model string/object for homogeneous agents or a list of `agents` for heterogeneous setups:

- `model` or `agents`: Model string/object for homogeneous agents, or list of agent models
- `num_agents`: Number of agents (default: 2)
- `tokenizer`: The tokenizer (required)
- `train_dataset`: Training dataset (required)
- `reward_func`: Callable that returns a list of floats (required)
- `reward_processor`: Optional processor to apply to rewards (e.g., scaling)
- `formatters`: Single callable or list of callables for each agent to format dataset items into prompts
- `external_transition`: Function providing transitions between turns (required for multi-turn training)
- `eval_dataset`: Evaluation dataset (optional)
- `eval_logger`: Evaluation logger function (optional)
- `eval_aggregator`: Evaluation aggregator function (optional)
- `wandb_config`: Configuration for Weights & Biases logging (optional)
- `model_config`: Model configuration dict (optional)
- `args`: Instance of `MAGRPOConfig` (optional)
{{% /hint %}}

{{% hint warning %}}
CoMLRL implements on-policy GRPO, which computes the policy gradient using the current policy's samples without importance sampling or ratio clipping.
{{% /hint %}}

{{% hint warning %}}
The trainer enforces `per_device_train_batch_size=1` and requires at least 2 generations for group baseline computation.
{{% /hint %}}

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
\Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} \Big( R^{(g)}_t - \max(R_t^{\mathcal{G}}) \Big) \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}\mid h_{i,t}) \Bigg].
{{< /katex >}}

{{% hint success %}}
These classes are derived from `comlrl.trainers.magrpo.MAGRPOTrainer`. Interfaces for the trainer and configuration classes are same as `MAGRPOTrainer` and `MAGRPOConfig`.
{{% /hint %}}
