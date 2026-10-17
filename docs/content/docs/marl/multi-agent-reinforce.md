---
title: Multi-Agent REINFORCE
weight: 1
aliases:
  - /docs/user-guide/multi-agent-reinforce/
math: true
---

REINFORCE is a class of policy-gradient methods that optimize the policy directly from sampled returns. Their simplicity and efficiency make them popular for LLM fine-tuning, as in [GRPO](https://arxiv.org/pdf/2402.03300), [Dr. GRPO](https://arxiv.org/abs/2503.20783), [RLOO](https://openreview.net/forum?id=r1lgTGL5DE), [ReMax](https://arxiv.org/abs/2310.10505), [TreeRPO](https://arxiv.org/abs/2506.05183), and [REINFORCE++](https://arxiv.org/abs/2501.03262). In multi-agent settings, several LLM agents respond at each turn, and their joint responses receive a shared reward.

All MAREINFORCE trainers share one training loop. At each rollout node, every agent samples {{< katex >}}K{{< /katex >}} responses, the trainer scores the index-aligned joint responses with one shared reward, and each agent weights its own log-probabilities by the resulting shared advantages before a buffered optimizer step.

{{< rollout-diagram diagram="reinforce-training" title="MAREINFORCE training flow" description="Every joint sample earns one shared return; its advantage weights each agent's own policy gradient." >}}

## MAREINFORCE

Before advantage normalization or reference regularization, the policy-gradient surrogate of Multi-Agent REINFORCE (MAREINFORCE) is:

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\sum_{t=0}^{H-1} R_t \cdot \log \pi_{\theta_i}(a_{i,t}\mid h_{i,t})\Bigg],
{{< /katex >}}
where {{< katex >}}R_t{{< /katex >}} is the return at turn {{< katex >}}t{{< /katex >}} and {{< katex >}}H{{< /katex >}} is the horizon (the number of dialog turns).
The expectation is taken over initial observations from the dataset {{< katex >}}\mathcal{D}{{< /katex >}} and the joint action history of all episodes following policy {{< katex >}}\boldsymbol{\pi}_{\boldsymbol{\theta}}{{< /katex >}}.

REINFORCE methods use no critic for value estimation, so their gradient estimates can have high variance from stochastic sampling and long-term credit assignment. Common remedies are an action-independent baseline and averaging over more samples, such as {{< katex >}}K{{< /katex >}} samples at each joint history {{< katex >}}\mathbf{h}_t{{< /katex >}}.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K} \sum_{t=0}^{H-1} \left(R^{k}_t - b(\mathbf{h}_t)\right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg],
{{< /katex >}}
where the baseline {{< katex >}}b(\mathbf{h}_t){{< /katex >}} is action-independent.

## MAGRPO

Multi-Agent Group-Relative Policy Optimization (MAGRPO) instantiates MAREINFORCE in the style of GRPO, using the mean return of the {{< katex >}}K{{< /katex >}} samples as its baseline. In the default `aligned` joint mode, its surrogate before advantage normalization is:

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K}\sum_{t=0}^{H-1} \left(R^{k}_t - \frac{1}{K}\sum_{l=1}^{K}R^{l}_t\right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg].
{{< /katex >}}

The sampled group mean includes the sample being updated, so it is not the action-independent baseline assumed in the general expression above. In multi-turn training, each branch's return is its immediate reward plus the mean return of its child node, with no discount applied. In `cross` mode, each agent's completion instead receives the **sum** of returns over all joint combinations containing that completion.

By default, `advantage_normalization=True` then centers and scales the advantages within each node's generation group using the population standard deviation, with a minimum denominator of `1e-6`. If reference KL regularization is enabled, the sampled reference penalty is subtracted from returns before the baseline and normalization are applied.

{{% hint warning %}}
The current `MAGRPOTrainer` loss slice omits the final completion token, so its implemented sequence score differs from the full sequence log-probability in the surrogate above. The canonical setting is `joint_mode="cross"`; the accepted `"crossed"` alias generates Cartesian combinations but does not apply the same per-completion return aggregation.
{{% /hint %}}

{{% hint info %}}
**MAGRPOConfig** parameters:

- `num_agents`: Configured number of agents, which should match the trainer setup
- `num_turns`: Number of turns per episode
- `num_train_epochs`: Number of training epochs
- `agent_learning_rate`: Learning rate
- `logging_steps`: Logging interval in steps
- `num_generations`: Number of generations to sample per prompt for each agent
- `max_new_tokens`: Maximum number of new tokens to generate
- `temperature`: Temperature for sampling
- `top_p`: Top-p for sampling
- `top_k`: Top-k for sampling
- `discount`: Retained configuration field; the current MAGRPO implementation computes undiscounted returns and does not use this value
- `joint_mode`: Joint action composition (`aligned` for index-aligned, `cross` for Cartesian product)
- `early_termination_threshold`: A training branch stops expanding when its node's mean immediate reward exceeds this threshold (default `-0.2`; `None` disables early termination)
- `rollout_buffer_size`: Number of node samples to buffer per agent before an update
- `train_batch_size`: Number of node samples per optimizer step; defaults to `rollout_buffer_size`
- `advantage_normalization`: Whether to center and scale advantages within each node's generation group (default `True`)
- `eval_interval`: Evaluation runs at batch indices divisible by N, including the first batch of each epoch; `0` disables periodic evaluation
- `eval_num_samples`: Number of samples to evaluate per evaluation run
- `eval_batch_size`: Evaluation DataLoader batch size
- `external_prompt_passthrough`: Whether to use external prompts directly in multi-turn training
- `advantage_mode`: Baseline mode (`mean`, `max`, `rloo`, `raw`)
- `parallel_training`: Agent execution mode (`none` or `mp`)
- `agent_devices`: Agent device placement; required explicitly for `parallel_training="mp"`
- `reference_kl_enabled`: Whether to enable a frozen-reference KL penalty (default `False`)
- `reference_kl_coef`: Weight of the reference penalty (default `0.1`)
- `reference_devices`: Optional device placement for the reference models
{{% /hint %}}

{{% hint info %}}
**MAGRPOTrainer** setup:

- `agent_model` or `agents`: Model identifier string for homogeneous agents, or a list of model identifiers or preloaded models; preloaded models are supplied through `agents`
- `num_agents`: Number of agents created from `agent_model` (default `2`), configured independently of `args`. When only `agents` is supplied, its length determines the agent count
- `tokenizer`: Shared tokenizer or one per agent; optional for model identifiers, required for preloaded models
- `reward_func`: Callable that returns a list of floats (required)
- `reward_processor`: Optional processor to apply to rewards (e.g., scaling)
- `formatters`: Single callable or list of per-agent callables for prompt formatting
- `args`: Instance of `MAGRPOConfig` (optional)
- `train_dataset`: Dataset required when calling `train()`
- `eval_dataset`: Evaluation dataset (optional)
- `model_config`: Model configuration dictionary (optional)
- `wandb_config`: Configuration for Weights & Biases logging (optional)
- `external_transition`: Function providing transitions between turns; required when `num_turns > 1` (the default is `2`)
- `eval_logger`: Optional task-metrics callback, called as `eval_logger(agent_completions_turns=..., batch_items=..., prompts=...)`; returns one metrics dictionary per evaluated item and is used with `eval_aggregator`
- `eval_aggregator`: Optional callback, called as `eval_aggregator(metrics, num_turns=...)`; the returned `turn_k/...` metrics are logged under `eval/`. Both evaluation callbacks must be supplied for custom metrics
{{% /hint %}}

{{% hint warning %}}
MAGRPO collects samples into a rollout buffer and recomputes their log-probabilities during updates, without importance sampling or ratio clipping. Samples may therefore predate earlier updates from the same buffer or rollout tree. Since the trainer has no critic model, `value_clip_range` does not apply.
{{% /hint %}}

{{% hint warning %}}
The trainer uses a fixed training DataLoader batch size of 1 and requires at least `num_generations=2`, including for the raw-return variant. By default, `train_batch_size=rollout_buffer_size`: each agent averages the losses of its buffered node samples for one optimizer step. Smaller mini-batches produce multiple optimizer steps; a partial buffer is flushed at the end of each epoch.
{{% /hint %}}

## Other Variants

CoMLRL also provides other MAREINFORCE variants with different baselines:

- **MARLOO**: Multi-Agent REINFORCE Leave-One-Out. The baseline is the mean return of the other generations in the same node's sample group, not the other agents.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K}\sum_{t=0}^{H-1} \left(R^{k}_t - \frac{1}{K-1}\sum_{l=1, l\neq k}^{K}R^{l}_t\right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg].
{{< /katex >}}

- **MAReMax**: Multi-Agent REINFORCE with Group Max (`MAReMaxConfig` and `MAReMaxTrainer`). The baseline is the maximum sampled group return at the node; this implementation does not generate a separate greedy baseline response.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t  \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\frac{1}{K}\sum_{k=1}^{K}\sum_{t=0}^{H-1} \left(R^{k}_t - \mathrm{max}_l\, R^l_t \right) \cdot \log \pi_{\theta_i}(a^{k}_{i,t}\mid h_{i,t})\Bigg].
{{< /katex >}}

{{% hint success %}}
`MAREINFORCETrainer`, `MARLOOTrainer`, and `MAReMaxTrainer` derive from `comlrl.trainers.reinforce.MAGRPOTrainer`. Their corresponding configuration classes inherit `MAGRPOConfig`, with default `advantage_mode` values of `raw`, `rloo`, and `max`, respectively. They share the same trainer interface and configuration fields.
{{% /hint %}}

The four trainers differ only in the baseline subtracted from the shared return. For one node with {{< katex >}}K=4{{< /katex >}} joint responses, the animation below compares their advantages before normalization: MAREINFORCE keeps the raw return, as in [REINFORCE](https://doi.org/10.1007/BF00992696); MAGRPO subtracts the group mean, as in GRPO; MARLOO subtracts the mean of the other {{< katex >}}K-1{{< /katex >}} returns, as in RLOO, which scales the MAGRPO advantages by {{< katex >}}K/(K-1){{< /katex >}}; and MAReMax subtracts the group maximum, so no response receives a positive advantage.

{{< rollout-diagram diagram="reinforce-baselines" title="Baselines across MAREINFORCE variants" description="Each row subtracts a different baseline from the same shared returns; shaded bars mark the advantages before normalization." >}}

{{% hint warning %}}
The variant formulas describe advantages **before normalization**. All variants inherit `advantage_normalization=True`, which re-centers even raw returns and max-baseline advantages. Except for the standard-deviation floor and numerical precision, the four modes then yield the same normalized advantages. Users can retain the baseline differences shown above, including the raw-return MAREINFORCE estimator, by setting `advantage_normalization=False`.
{{% /hint %}}
