---
title: Multi-Agent Actor-Critic
weight: 2
aliases:
  - /docs/user-guide/multi-agent-actor-critic/
math: true
---

Actor-Critic (AC) methods are widely used policy-gradient methods that train a critic to support the policy. They can reach lower variance and better sample efficiency than REINFORCE, but the critic needs careful design and tuning for stable training. In Multi-Agent Reinforcement Learning (MARL), AC methods include Multi-Agent Actor-Critic (MAAC) and Independent Actor-Critic (IAC).

Both roll out the agents jointly and score their joint response with the reward function, but they use critics differently. MAAC trains one critic on the joint history, so every agent shares its TD error. IAC trains one critic per agent on that agent's history, so each agent gets its own TD error.

{{< rollout-diagram diagram="ac-training" title="Actor-critic training flow" description="MAAC's centralized critic gives all agents the same TD error; IAC's independent critics give each agent its own." >}}

## MAAC

Multi-Agent Actor-Critic (MAAC) uses a Centralized Critic (CC) shared by all agents to evaluate joint histories {{< katex >}}V_{\boldsymbol{\phi}}(\mathbf{h}_t){{< /katex >}} with `critic_type="v"`, or joint history-action pairs {{< katex >}}Q_{\boldsymbol{\psi}}(\mathbf{h}_t, \mathbf{a}_t){{< /katex >}} with `critic_type="q"`. The critic reads the agents' prompts joined into one text sequence, plus their completions in Q mode. Each agent's policy gradient is:

{{< katex display=true >}}
\nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\boldsymbol{\pi}}\left[\sum_{t=0}^{H-1} \nabla_{\theta_i} \log \pi_{\theta_i}(a_{i,t}|h_{i,t})\,\boldsymbol{\delta}_t\right]
{{< /katex >}}

where {{< katex inline=true >}}\boldsymbol{\delta}_t = r_t + \gamma V_{\boldsymbol{\phi}}(\mathbf{h}_{t+1}) - V_{\boldsymbol{\phi}}(\mathbf{h}_{t}){{< /katex >}}, and the critic is updated by:

{{< katex display=true >}}
\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{\boldsymbol{\pi}}\left[\sum_{t=0}^{H-1} \big(r_t + \gamma V_{\phi}(\mathbf{h}_{t+1}) - V_{\phi}(\mathbf{h}_{t})\big)^2\right].
{{< /katex >}}

Here {{< katex >}}r_t{{< /katex >}} is the processed reward, minus the reference-KL penalty when that is enabled; rewards may be shared across agents or given per agent. The last collected turn, including an early-terminated or single-turn rollout, uses the reward alone as its target, without bootstrapping. A completion's log-probability sums its generated tokens, excluding prompt and trailing padding tokens, and the loss averages over samples. Before updating, buffers are grouped by agent and turn, and `advantage_normalization` normalizes the TD errors within each group; targets and advantages stay fixed during the update. The shared critic is updated with each agent's batch in turn, and `value_loss_coef` scales its loss.

Q mode uses the same TD-error actor weight and bootstraps with a sampled next action; it does not implement an actor objective weighted directly by Q or a state-only baseline for Q.

{{% hint info %}}
**MAACConfig** parameters:
- `num_agents`: Number of agents
- `num_turns`: Number of turns
- `critic_type`: Critic target type (`v` for V(h), `q` for Q(h,a))
- `num_train_epochs`: Number of training epochs
- `agent_learning_rate`: Learning rate for agents
- `critic_learning_rate`: Learning rate for the shared critic
- `value_loss_coef`: Weight on critic loss
- `advantage_normalization`: Whether to normalize advantages before updates
- `rollout_buffer_size`: Number of samples to collect per agent before an update
- `train_batch_size`: Mini-batch size within each update
- `max_new_tokens`: Maximum tokens to generate per completion
- `temperature`: Temperature for sampling
- `top_p`: Top-p for nucleus sampling
- `top_k`: Top-k for sampling
- `num_generations`: Number of generations per prompt per agent
- `external_prompt_passthrough`: Whether to use external prompts directly in multi-turn training
- `discount`: Discount factor for the next-turn critic prediction in TD targets; logged `expected_return` uses undiscounted reward sums
- `early_termination_threshold`: A rollout ends after a turn whose mean processed reward exceeds this threshold; `None` disables early stopping
- `eval_interval`: Evaluation interval (in training batches)
- `eval_num_samples`: Number of evaluation samples per interval
- `eval_batch_size`: Evaluation DataLoader batch size
- `logging_steps`: Minimum number of joint-action reward evaluations between training metric logs
- `parallel_training`: `none` for one device or `mp` for explicit agent/critic device placement
- `agent_devices`, `critic_devices`: Device assignments, required in `mp` mode
- `reference_kl_enabled`, `reference_kl_coef`: Whether the frozen-reference KL reward penalty is enabled, and its weight (disabled by default)
- `reference_devices`: Optional device placement for reference models
{{% /hint %}}

{{% hint info %}}
**MAACTrainer** setup:

- `agent_model` or `agents`: Actor model identifier string for homogeneous agents, or list of agent models (multi-agent `agent_model` must be a string)
- `critic_model` or `critics`: Required single shared critic, supplied as an identifier, model instance, or 1-element list
- `tokenizer`: Shared tokenizer, or a per-agent list when using `agents`; required with `agents` or a model instance, otherwise loaded from `agent_model`
- `reward_func`: Callable returning rewards (required)
- `reward_processor`: Optional reward post-processor
- `formatters`: Single callable or list for per-agent prompt formatting
- `args`: Instance of `MAACConfig` (optional)
- `train_dataset`: Training dataset (required)
- `eval_dataset`: Optional evaluation dataset
- `model_config`: Model loading options; `critic_value_head_hidden_dim` configures the centralized critic's value head (optional)
- `wandb_config`: Weights & Biases logging configuration (optional)
- `metrics_callback`: Optional callback taking rollout samples and returning a dictionary of scalar metrics
- `external_transition`: Transition function required when `num_turns > 1`
{{% /hint %}}

{{% hint warning %}}
MAAC updates on collected samples without importance sampling or policy-ratio clipping. `MAACConfig` has no `value_clip_range` option. The shared critic uses the first agent's tokenizer, which must be compatible with the critic model.
{{% /hint %}}

{{% hint warning %}}
The trainer uses a fixed training DataLoader batch size of 1. For `num_turns > 1`, an `external_transition` and `num_generations=1` are required. `train_batch_size` defaults to `rollout_buffer_size`; actual updates use the available samples in each agent/turn group, including smaller groups when buffers are flushed.
{{% /hint %}}

## IAC

Independent Actor-Critic (IAC) optimizes each agent's policy independently, using returns from the agents' joint responses. Each agent has its own actor and critic and treats the other agents as part of the environment. Rewards can be shared or agent-specific. Each critic conditions on its agent's prompt and history in V mode, or on its prompt and completion in Q mode. The policy gradient for each agent is:

{{< katex display=true >}}
\nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\boldsymbol{\pi}}\left[\sum_{t=0}^{H-1} \nabla_{\theta_i} \log \pi_{\theta_i}(a_{i,t}|h_{i,t})\,\delta_{i,t}\right]
{{< /katex >}}

where {{< katex inline=true >}}\delta_{i,t} = r_t + \gamma V_{\phi_i}(h_{i,t+1}) - V_{\phi_i}(h_{i,t}){{< /katex >}}. The actor uses the same sample-averaged loss and completion-token sum as MAAC, with optional normalization within each agent/turn group. The Q-mode actor also uses this TD error.

CoMLRL supports two IAC variants:

- **Separate Critic**: Uses an independent model for value estimation. Actor and critic parameters are optimized separately, at the cost of storing an additional model per agent.

- **Shared Model**: Attaches a small value head to the transformer backbone, sharing the actor model's history (or history-action) representations to reduce memory use.

The critics are updated by minimizing the TD error:

{{< katex display=true >}}
\mathcal{L}(\phi_i) = \mathbb{E}_{\boldsymbol{\pi}}\left[\sum_{t=0}^{H-1} \big(r_t + \gamma V_{\phi_i}(h_{i,t+1}) - V_{\phi_i}(h_{i,t})\big)^2\right].
{{< /katex >}}

When using a shared model (`use_separate_critic=False`), the actor and value head optimize the actor loss plus the critic loss weighted by `value_loss_coef`. Unless `value_clip_range=None`, a value clip is applied to improve training stability:

{{< katex display=true >}}
L(\phi_i) = \max\Big( (V_{\phi_i}(h_t) - \hat{V}_t)^2,\ (V_{\phi_i}^{\text{clip}}(h_t) - \hat{V}_t)^2 \Big)
\\ V_{\phi_i}^{\text{clip}}(h_t) = V_{\phi_i}^{\text{old}}(h_t) + \mathrm{clip}(V_{\phi_i}(h_t) - V_{\phi_i}^{\text{old}}(h_t), -\epsilon, \epsilon),
{{< /katex >}}

where {{< katex >}}\hat{V}_t{{< /katex >}} is the TD target, {{< katex >}}\epsilon{{< /katex >}} is `value_clip_range`, and the errors are averaged across the mini-batch. Separate critics use the unclipped squared error.

{{% hint info %}}
**IACConfig** parameters:

- `num_agents`: Number of agents
- `num_turns`: Number of turns
- `num_train_epochs`: Number of training epochs
- `agent_learning_rate`: Learning rate for agents
- `critic_learning_rate`: Learning rate for separate critics; shared models use `agent_learning_rate` for both actor and value head
- `value_loss_coef`: Coefficient for value loss
- `value_clip_range`: Value clipping range in shared-model mode; `None` disables clipping
- `advantage_normalization`: Whether to normalize advantages
- `rollout_buffer_size`: Number of samples to collect before an update
- `train_batch_size`: Mini-batch size for policy updates
- `max_new_tokens`: Maximum new tokens to generate
- `temperature`: Temperature for sampling
- `top_p`: Top-p for nucleus sampling
- `top_k`: Top-k for sampling
- `num_generations`: Number of generations per prompt per agent
- `use_separate_critic`: Whether to use a separate critic model
- `critic_type`: Critic target type (`v` for V(h), `q` for Q(h,a))
- `critic_value_head_hidden_dim`: Hidden dimension for critic value head
- `value_head_hidden_dim`: Hidden dimension for value head in shared-model mode
- `external_prompt_passthrough`: Whether to use external prompts directly in multi-turn training
- `discount`: Discount factor in TD targets; logged `expected_return` uses undiscounted reward sums
- `early_termination_threshold`: A rollout ends when a turn's mean processed reward exceeds this threshold; `None` disables early stopping
- `eval_interval`: Evaluation interval (in training batches)
- `eval_num_samples`: Number of evaluation samples per interval
- `eval_batch_size`: Evaluation DataLoader batch size
- `logging_steps`: Minimum number of joint-action reward evaluations between training metric logs
- `parallel_training`: `none` for one device or `mp` for explicit agent/critic device placement
- `agent_devices`, `critic_devices`: Device assignments; `mp` requires `agent_devices` and, for separate critics, `critic_devices`
- `reference_kl_enabled`, `reference_kl_coef`: Whether the frozen-reference KL reward penalty is enabled, and its weight (disabled by default)
- `reference_devices`: Optional device placement for reference models
{{% /hint %}}

{{% hint info %}}
**IACTrainer** setup:

- `agent_model` or `agents`: Model identifier string for homogeneous agents, or list of agent models (multi-agent `agent_model` must be a string)
- `critic_model` or `critics`: Critic identifier, model instance, or list of critic models when `use_separate_critic=True`
- `tokenizer`: Shared tokenizer, or a per-agent list when using `agents`; required with `agents` or a model instance, otherwise loaded from `agent_model`
- `reward_func`: Callable that returns a list of floats (required)
- `reward_processor`: Optional processor to apply to rewards
- `formatters`: Single callable or list of per-agent callables for formatting dataset items into prompts
- `args`: Instance of `IACConfig` (optional)
- `train_dataset`: Training dataset (required)
- `eval_dataset`: Evaluation dataset (optional)
- `model_config`: Model configuration dictionary (optional)
- `wandb_config`: Configuration for Weights & Biases logging (optional)
- `metrics_callback`: Optional callback taking rollout samples and returning a dictionary of scalar metrics
- `external_transition`: Optional transition function required for multi-turn training
{{% /hint %}}

{{% hint warning %}}
IAC updates on collected samples without importance sampling or policy-ratio clipping. In shared-model mode (`use_separate_critic=False`), value heads are attached to the actor models; passing `critic_model` or `critics` raises an error, and `value_clip_range` only applies in this mode. In separate-critic mode (`use_separate_critic=True`), users can supply a `critics` list of length `num_agents` or a single `critic_model` source to broadcast. Distinct model instances, or an identifier loaded separately for each critic, keep the critics' parameters independent; broadcasting one preloaded model object shares its parameters. Each critic consumes its actor's token IDs, so a different critic model must be compatible with that actor's tokenizer. Agents may be homogeneous or heterogeneous in either mode.
{{% /hint %}}

{{% hint warning %}}
The trainer uses a fixed training DataLoader batch size of 1. For `num_turns > 1`, an `external_transition` and `num_generations=1` are required. `train_batch_size` defaults to `rollout_buffer_size`; actual updates use the available samples in each agent/turn group, including smaller groups when buffers are flushed.
{{% /hint %}}
