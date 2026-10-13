---
title: Multi-Agent REINFORCE
weight: 1
aliases:
  - /docs/user-guide/multi-agent-reinforce/
math: true
---

REINFORCE is a class of policy gradient methods that optimize the policy directly using sampled returns.
It has been widely used to fine-tune LLMs because of its simplicity and efficiency, e.g., [GRPO](https://arxiv.org/pdf/2402.03300), [Dr. GRPO](https://arxiv.org/abs/2503.20783), [RLOO](https://openreview.net/forum?id=r1lgTGL5DE), [ReMax](https://arxiv.org/abs/2310.1050), [TreeRPO](https://arxiv.org/abs/2506.05183), and [REINFORCE++](https://arxiv.org/abs/2501.03262).
REINFORCE can be extended to multi-agent settings, where multiple LLM agents respond at each turn and their joint responses receive a shared reward.

## MA-REINFORCE

The policy-gradient surrogate for Multi‑Agent REINFORCE (MA-REINFORCE), before advantage normalization or reference regularization, can be expressed as:

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\sum_{t=0}^{H-1} R_t \cdot \log \pi_{\theta_i}(a_{i,t}\mid h_{i,t})\Bigg],
{{< /katex >}}
where {{< katex >}}R_t{{< /katex >}} is the return at turn {{< katex >}}t{{< /katex >}} and {{< katex >}}H{{< /katex >}} is the horizon (i.e., number of dialog turns).
The expectation is taken over initial observations from the dataset {{< katex >}}\mathcal{D}{{< /katex >}} and the joint action history of all episodes following policy {{< katex >}}\boldsymbol{\pi}_{\boldsymbol{\theta}}{{< /katex >}}.

REINFORCE methods do not use a critic model for value estimation. Their policy-gradient estimates can have high variance due to stochastic sampling and long-term credit assignment.
Common ways to reduce this variance include using an action-independent baseline or averaging over more samples, e.g., {{< katex >}}K{{< /katex >}} samples at each joint history {{< katex >}}\mathbf{h}_t{{< /katex >}}.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K} \sum_{t=0}^{H-1} \left(R^{k}_t - b(\mathbf{h}_t)\right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg],
{{< /katex >}}
where the baseline {{< katex >}}b(\mathbf{h}_t){{< /katex >}} is action-independent.

## MAGRPO

Multi‑Agent Group‑Relative Policy Optimization (MAGRPO) is an instantiation of MA-REINFORCE inspired by GRPO, where the group-average baseline is the mean return of {{< katex >}}K{{< /katex >}} samples. In the default `aligned` joint mode, its surrogate before advantage normalization is:

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K}\sum_{t=0}^{H-1} \left(R^{k}_t - \frac{1}{K}\sum_{l=1}^{K}R^{l}_t\right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg].
{{< /katex >}}

The sampled group mean includes the sample being updated, so it is not the action-independent baseline assumed in the general expression above. In multi-turn training, each branch's return is its immediate reward plus the mean return of its child node, with no discount applied. In `cross` mode, each agent's completion instead receives the **sum** of returns over all joint combinations containing that completion.

By default, `advantage_normalization=True` then centers and scales the advantages within each node's generation group using the population standard deviation, with a minimum denominator of `1e-6`. If reference KL regularization is enabled, the sampled reference penalty is subtracted from returns before the baseline and normalization are applied.

{{% hint warning %}}
The current `MAGRPOTrainer` loss slice omits the final completion token, so its implemented sequence score differs from the full sequence log-probability in the surrogate above. Use the canonical `joint_mode="cross"`: the accepted `"crossed"` alias generates Cartesian combinations but does not apply the same per-completion return aggregation.
{{% /hint %}}

{{% hint info %}}
**MAGRPOConfig** parameters:

- `num_agents`: Configured number of agents; keep this consistent with the trainer setup
- `num_turns`: Number of turns per episode
- `num_train_epochs`: Number of training epochs
- `agent_learning_rate`: Learning rate
- `logging_steps`: Log every N steps
- `num_generations`: Number of generations to sample per prompt for each agent
- `max_new_tokens`: Maximum number of new tokens to generate
- `temperature`: Temperature for sampling
- `top_p`: Top-p for sampling
- `top_k`: Top-k for sampling
- `discount`: Retained configuration field; the current MAGRPO implementation computes undiscounted returns and does not use this value
- `joint_mode`: Joint action composition (`aligned` for index-aligned, `cross` for Cartesian product)
- `early_termination_threshold`: Stop expanding a training branch when its node's mean immediate reward exceeds this threshold (default `-0.2`; `None` disables early termination)
- `rollout_buffer_size`: Number of node samples to buffer per agent before an update
- `train_batch_size`: Number of node samples per optimizer step; defaults to `rollout_buffer_size`
- `advantage_normalization`: Center and scale advantages within each node's generation group (default `True`)
- `eval_interval`: Run evaluation at batch indices divisible by N, including the first batch of each epoch; `0` disables periodic evaluation
- `eval_num_samples`: Number of samples to evaluate per evaluation run
- `eval_batch_size`: Eval dataloader batch size
- `external_prompt_passthrough`: Use external prompts directly in multi-turn
- `advantage_mode`: Baseline mode (`mean`, `max`, `rloo`, `raw`)
- `parallel_training`: Agent execution mode (`none` or `mp`)
- `agent_devices`: Agent device placement; required explicitly for `parallel_training="mp"`
- `reference_kl_enabled`: Enable a frozen-reference KL penalty (default `False`)
- `reference_kl_coef`: Weight of the reference penalty (default `0.1`)
- `reference_devices`: Optional device placement for the reference models
{{% /hint %}}

{{% hint info %}}
**MAGRPOTrainer** setup:

- `agent_model` or `agents`: Model identifier string for homogeneous agents, or a list of model identifiers or preloaded models; use `agents` for preloaded models
- `num_agents`: Number of agents created from `agent_model` (default `2`); pass this explicitly even when it is set in `args`. When only `agents` is supplied, its length determines the agent count
- `tokenizer`: Shared tokenizer or one per agent; optional for model identifiers, required for preloaded models
- `reward_func`: Callable that returns a list of floats (required)
- `reward_processor`: Optional processor to apply to rewards (e.g., scaling)
- `formatters`: Single callable or list of callables for each agent to format prompts
- `args`: Instance of `MAGRPOConfig` (optional)
- `train_dataset`: Dataset required when calling `train()`
- `eval_dataset`: Evaluation dataset (optional)
- `model_config`: Model configuration dict (optional)
- `wandb_config`: Configuration for Weights & Biases logging (optional)
- `external_transition`: Function providing transitions between turns; required when `num_turns > 1` (the default is `2`)
- `eval_logger`: Optional task metrics, called as `eval_logger(agent_completions_turns=..., batch_items=..., prompts=...)`; returns one metrics dict per evaluated item. Used together with `eval_aggregator`
- `eval_aggregator`: Optional, called as `eval_aggregator(metrics, num_turns=...)`; the returned `turn_k/...` metrics are logged under `eval/`. Both evaluation callbacks must be supplied for custom metrics
{{% /hint %}}

{{% hint warning %}}
MAGRPO collects samples into a rollout buffer and recomputes their log probabilities during updates, without importance sampling or ratio clipping. Samples may therefore predate earlier updates from the same buffer or rollout tree. Since the trainer has no critic model, `value_clip_range` does not apply.
{{% /hint %}}

{{% hint warning %}}
The trainer uses a fixed training DataLoader batch size of 1 and requires at least `num_generations=2`, including for the raw-return variant. By default, `train_batch_size=rollout_buffer_size`: each agent averages the losses of its buffered node samples for one optimizer step. Smaller mini-batches produce multiple optimizer steps; a partial buffer is flushed at the end of each epoch.
{{% /hint %}}

## Other Variants

CoMLRL also provides other MA-REINFORCE variants with different baselines:

- **MARLOO**: Multi‑Agent REINFORCE Leave‑One‑Out. The baseline is the mean return of the other generations in the same node's sample group, not the other agents.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K}\sum_{t=0}^{H-1} \left(R^{k}_t - \frac{1}{K-1}\sum_{l=1, l\neq k}^{K}R^{l}_t\right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg].
{{< /katex >}}

- **MAReMax**: Multi‑Agent REINFORCE with Group Max (`MAReMaxConfig` and `MAReMaxTrainer`). The baseline is the maximum sampled group return at the node; this implementation does not generate a separate greedy baseline response.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K}\sum_{t=0}^{H-1} \left(R^{k}_t - \mathrm{max}_l\, R^l_t \right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg].
{{< /katex >}}

{{% hint success %}}
`MAREINFORCETrainer`, `MARLOOTrainer`, and `MAReMaxTrainer` derive from `comlrl.trainers.reinforce.MAGRPOTrainer`. Their corresponding configuration classes inherit `MAGRPOConfig`, with default `advantage_mode` values of `raw`, `rloo`, and `max` respectively. They share the same trainer interface and configuration fields.
{{% /hint %}}

{{% hint warning %}}
The variant formulas describe advantages **before normalization**. All variants inherit `advantage_normalization=True`, which re-centers even raw returns and max-baseline advantages. Except for the standard-deviation floor and numerical precision, the four modes then yield the same normalized advantages. Set `advantage_normalization=False` to retain the baseline differences shown above, including the raw-return MA-REINFORCE estimator.
{{% /hint %}}
