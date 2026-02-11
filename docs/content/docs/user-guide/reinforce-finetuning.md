---
title: Multi-Agent REINFORCE
weight: 2
math: true
---

REINFORCE is a class of policy gradient methods that optimize the policy directly using sampled returns.
It has been widely used to fine-tune LLMs because of its simplicity and efficiency, e.g., [GRPO](https://arxiv.org/pdf/2402.03300), [Dr. GRPO](https://arxiv.org/abs/2503.20783), [RLOO](https://openreview.net/forum?id=r1lgTGL5DE), [ReMax](https://arxiv.org/abs/2310.1050), [TreeRPO](https://arxiv.org/abs/2506.05183), and [REINFORCE++](https://arxiv.org/abs/2501.03262).
REINFORCE can be extended to multi-agent settings, where multiple LLM agents response synchronously and their joint responses form a solution at each turn to receive a shared reward at each turn.

## MA-REINFORCE

The naive Multi‑Agent REINFORCE (MA-REINFORCE) can be expressed as:

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\sum_{t=0}^{H-1} R_t \cdot \log \pi_{\theta_i}(a_{i,t}\mid h_{i,t})\Bigg],
{{< /katex >}}
where {{< katex >}}R_t{{< /katex >}} is the return at turn {{< katex >}}t{{< /katex >}} and {{< katex >}}H{{< /katex >}} is the horizon (i.e., number of dialog turns).
The expectation is taken over initial observations from the dataset {{< katex >}}\mathcal{D}{{< /katex >}} and the joint action history of all episodes following policy {{< katex >}}\boldsymbol{\pi}_{\boldsymbol{\theta}}{{< /katex >}}.

REINFORCE methods do not use a critic model for value estimation. Their policy gradients estimation can have high variance, due to the stochasticity of the environment and the long-term credit assignment.
There are two common approaches to reduce the variance: using an action-independent baseline or update with more samples, e.g., using {{< katex >}}K{{< /katex >}} samples for value estimation of each joint history {{< katex >}}\mathbf{h}_t{{< /katex >}}.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K} \sum_{t=0}^{H-1} \left(R^{k}_t - b(\mathbf{h}_t)\right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg],
{{< /katex >}}
where the baseline {{< katex >}}b(\mathbf{h}_t){{< /katex >}} is action-independent.

## MAGRPO

Multi‑Agent Group‑Relative Policy Optimization (MAGRPO) is an instantiation of MA-REINFORCE inspired from GRPO, where the group-average baseline is the mean return of {{< katex >}}K{{< /katex >}} samples:

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K}\sum_{t=0}^{H-1} \left(R^{k}_t - \frac{1}{K}\sum_{l=1}^{K}R^{l}_t\right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg].
{{< /katex >}}

{{% hint info %}}
**MAGRPOConfig** parameters:

- `num_agents`: Number of agents (default: 2)
- `num_turns`: Number of turns per episode (default: 2)
- `num_train_epochs`: Number of training epochs (default: 20)
- `agent_learning_rate`: Learning rate (default: 5e-6)
- `logging_steps`: Log every N steps (default: 50)
- `num_generations`: Number of generations to sample per prompt for each agent (default: 4)
- `max_new_tokens`: Maximum number of new tokens to generate (default: 256)
- `temperature`: Temperature for sampling (default: 0.6)
- `top_p`: Top-p for sampling (default: 0.6)
- `top_k`: Top-k for sampling (default: 50)
- `discount`: Discount factor gamma over turns for returns (default: 0.9)
- `joint_mode`: Joint action composition
  - `'aligned'` (index-aligned, default)
  - `'cross'` (Cartesian product)
- `early_termination_threshold`: Stop rollouts with mean reward exceeds a threshold (default: -0.2)
- `rollout_buffer_size`: Number of node samples to buffer before update (default: 2)
- `train_batch_size`: Mini-batch size within each update (default: rollout_buffer_size)
- `advantage_normalization`: Whether to normalize advantages (default: true)
- `eval_interval`: Run evaluation every N training batches (default: 16)
- `eval_num_samples`: Number of samples to evaluate per evaluation run (default: 4)
- `eval_batch_size`: Eval dataloader batch size (default: 1)
- `external_prompt_passthrough`: Use external prompts directly in multi-turn (default: false)
- `advantage_mode`: Baseline mode (`mean`, `max`, `rloo`, `raw`) (default: mean)
{{% /hint %}}

{{% hint info %}}
**MAGRPOTrainer** accepts either an `agent_model` string for homogeneous agents or a list of `agents` for heterogeneous setups. When both provided, `agents` overrides `agent_model` and must be a list of model identifiers (e.g., Hugging Face model names).

- `agent_model` or `agents`: Model identifier string for homogeneous agents, or list of agent models (multi-agent `agent_model` must be a string)
- `num_agents`: Number of agents (default: 2)
- `tokenizer`: The tokenizer (required)
- `train_dataset`: Training dataset (required)
- `reward_func`: Callable that returns a list of floats (required)
- `reward_processor`: Optional processor to apply to rewards (e.g., scaling)
- `formatters`: Single callable or list of callables for each agent to format prompts
- `external_transition`: Function providing transitions between turns
- `eval_dataset`: Evaluation dataset (optional)
- `eval_logger`: Evaluation logger function (optional)
- `eval_aggregator`: Evaluation aggregator function (optional)
- `wandb_config`: Configuration for Weights & Biases logging (optional)
- `model_config`: Model configuration dict (optional)
- `args`: Instance of `MAGRPOConfig` (optional)
{{% /hint %}}

{{% hint warning %}}
For simplicity, MAGRPO computes the policy gradient using the current policy's samples without importance sampling or ratio clipping. And since it does not use a critic model, there is no `value_clip_range` applicable.
{{% /hint %}}

{{% hint warning %}}
The trainer uses a fixed training DataLoader batch size of 1 and requires at least `num_generations=2` generations for group baseline computation. The training use batch gradient descent by default, where `train_batch_size`=`rollout_buffer_size`.
{{% /hint %}}

## Other Variants

CoMLRL also provides other MA-REINFORCE variants with different baselines:

- **MARLOO**: Multi‑Agent REINFORCE Leave‑One‑Out. Baseline is the mean return of other agents (leave‑one‑out) at the same step.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K}\sum_{t=0}^{H-1} \left(R^{k}_t - \frac{1}{K-1}\sum_{l=1, l\neq k}^{K}R^{l}_t\right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg].
{{< /katex >}}

- **MAREMAX**: Multi‑Agent REINFORCE with Group Max. Baseline is the maximum group return at the step.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K}\sum_{t=0}^{H-1} \left(R^{k}_t - \mathrm{max}_l\, R^l_t \right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg].
{{< /katex >}}

{{% hint success %}}
These classes and MA-REINFORCE are derived from `comlrl.trainers.reinforce.MAGRPOTrainer`. Interfaces for the trainer and configuration classes are the same as `MAGRPOTrainer` and `MAGRPOConfig`.
{{% /hint %}}
