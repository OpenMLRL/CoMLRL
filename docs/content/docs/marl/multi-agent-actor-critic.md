---
title: Multi-Agent Actor-Critic
weight: 2
aliases:
  - /docs/user-guide/multi-agent-actor-critic/
math: true
---

Actor-Critic (AC) methods are widely used policy gradient methods that use critics to support training.
AC methods can achieve lower variance and better sample efficiency than REINFORCE, but this requires careful design and tuning of the critic to ensure stable training.
In Multi-Agent Reinforcement Learning (MARL), Actor-Critic methods include Multi-Agent Actor-Critic (MAAC) and Independent Actor-Critic (IAC).

## MAAC

Multi-Agent Actor-Critic (MAAC) uses one Centralized Critic (CC) across agents to evaluate joint histories with `critic_type="v"`, or joint history-action pairs with `critic_type="q"`. The critic receives the agents' prompts joined into a single text sequence; Q mode also includes their generated completions.

The critic prediction is denoted by {{< katex >}}C_{\boldsymbol{\phi}}(x_t){{< /katex >}}, where {{< katex >}}x_t=\mathbf{h}_t{{< /katex >}} in V mode and {{< katex >}}x_t=(\mathbf{h}_t,\mathbf{a}_t){{< /katex >}} in Q mode. CoMLRL forms a one-step target and a TD residual for each agent:

{{< katex display=true >}}
y_{i,t}=\tilde r_{i,t}+\gamma m_t\bar C(x_{t+1}),
\qquad A_{i,t}=y_{i,t}-\bar C(x_t),
{{< /katex >}}

where {{< katex >}}\bar C{{< /katex >}} is the detached prediction recorded during rollout collection, and {{< katex >}}m_t{{< /katex >}} is 1 only when another turn was collected. The final collected turn, including an early-terminated or single-turn rollout, uses {{< katex >}}y_{i,t}=\tilde r_{i,t}{{< /katex >}} without bootstrapping. Rewards may be shared across agents or supplied per agent. Here {{< katex >}}\tilde r_{i,t}{{< /katex >}} denotes the processed reward, with the optional reference-KL penalty subtracted when enabled.

The actor and critic minimize the following losses over each agent's mini-batch {{< katex >}}B_i{{< /katex >}}:

{{< katex display=true >}}
\mathcal L_{\pi_i}=-\frac{1}{|B_i|}\sum_{t\in B_i}\log\pi_{\theta_i}(a_{i,t}\mid h_{i,t})\,\operatorname{sg}(\widehat A_{i,t}),
\qquad
\mathcal L_C=\frac{1}{|B_i|}\sum_{t\in B_i}\big(C_{\boldsymbol{\phi}}(x_t)-y_{i,t}\big)^2.
{{< /katex >}}

Here {{< katex >}}\widehat A{{< /katex >}} is the optionally normalized residual and {{< katex >}}\operatorname{sg}{{< /katex >}} means stop-gradient. The completion log-probability is the sum of generated-token log-probabilities; prompt and trailing padding tokens are excluded. The loss then averages across samples. Buffers are grouped by agent and turn before updating, and `advantage_normalization` normalizes the residuals within each such group. Targets and advantages remain fixed during the update. The shared critic is updated sequentially with each agent's batch, and `value_loss_coef` scales its loss.

The animation below follows one MAAC turn from sampling through the update for two agents. Both actors use residuals from the same centralized critic, which is then trained against the TD targets using each agent's batch in turn; IAC instead gives each agent its own critic.

{{< rollout-diagram diagram="ac-training" title="MAAC training step" description="Each agent samples one response per turn; the centralized critic turns the reward and next history into TD targets and residuals that weight each actor's loss and regress the critic." >}}

In Q mode, the implementation uses the same TD-residual actor weight, with a sampled next action in the bootstrap. It does not implement a separate actor objective weighted directly by Q or a state-only baseline for Q.

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

Independent Actor-Critic (IAC) optimizes each agent's policy with its own critic; other agents serve as part of the environment. Rewards can be shared or agent-specific. The critic conditions on that agent's prompt/history in V mode, or on its prompt and completion in Q mode. IAC uses the same detached-target convention as MAAC:

{{< katex display=true >}}
y_{i,t}=\tilde r_{i,t}+\gamma m_t\bar C_i(x_{i,t+1}),
\qquad A_{i,t}=y_{i,t}-\bar C_i(x_{i,t}),
{{< /katex >}}

where {{< katex >}}x_{i,t}=h_{i,t}{{< /katex >}} in V mode and {{< katex >}}x_{i,t}=(h_{i,t},a_{i,t}){{< /katex >}} in Q mode. The actor uses the same sample-averaged loss and completion-token sum as MAAC, with optional normalization within each agent/turn group. The Q-mode actor also uses this TD residual.

CoMLRL supports two IAC variants:

- **Separate Critic**: Uses an independent model for value estimation. Actor and critic parameters are optimized separately, at the cost of storing an additional model per agent.

- **Shared Model**: Attaches a small value head to the transformer backbone, sharing the actor model's history (or history-action) representations to reduce space requirements.

The critics are updated by minimizing the TD error:

{{< katex display=true >}}
\mathcal L_{C_i}=\frac{1}{|B_i|}\sum_{t\in B_i}\big(C_{\phi_i}(x_{i,t})-y_{i,t}\big)^2.
{{< /katex >}}

When using a shared model (`use_separate_critic=False`), the actor and value head optimize the combined loss {{< katex >}}\mathcal L_{\pi_i}+\texttt{value\_loss\_coef}\,\mathcal L_{C_i}{{< /katex >}}. Unless `value_clip_range=None`, the value loss takes the larger error from the current prediction and a prediction clipped relative to the rollout value, for either critic type:

{{< katex display=true >}}
\ell_{C_i}=\max\Big((C_{\phi_i}(x_{i,t})-y_{i,t})^2,\ (C_{\phi_i}^{\mathrm{clip}}(x_{i,t})-y_{i,t})^2\Big),
\\ C_{\phi_i}^{\mathrm{clip}}(x)=\bar C_i(x)+\operatorname{clip}\big(C_{\phi_i}(x)-\bar C_i(x),-\epsilon,\epsilon\big),
{{< /katex >}}

where {{< katex >}}\epsilon{{< /katex >}} is `value_clip_range` and these errors are averaged across the mini-batch. Separate critics use the unclipped squared error.

{{% hint info %}}
**IACConfig** provides parameters for configuring Independent Actor-Critic training:

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
**IACTrainer** trains agents using Independent Actor-Critic:

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
IAC updates on collected samples without importance sampling or policy-ratio clipping. In shared-model mode (`use_separate_critic=False`), value heads are attached to the actor models; passing `critic_model` or `critics` raises an error, and `value_clip_range` only applies in this mode. Agents may be homogeneous or heterogeneous. In separate-critic mode (`use_separate_critic=True`), users can supply a `critics` list of length `num_agents` or a single `critic_model` source for broadcasting. Distinct model instances or an identifier loaded separately for each critic keep their parameters independent; broadcasting one preloaded model object reuses its parameters. Each critic consumes its actor's token IDs, so a different critic model must be compatible with that actor's tokenizer.
{{% /hint %}}

{{% hint warning %}}
The trainer uses a fixed training DataLoader batch size of 1. For `num_turns > 1`, an `external_transition` and `num_generations=1` are required. `train_batch_size` defaults to `rollout_buffer_size`; actual updates use the available samples in each agent/turn group, including smaller groups when buffers are flushed.
{{% /hint %}}
