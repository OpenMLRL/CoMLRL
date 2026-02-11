---
title: Multi-Agent Actor-Critic
weight: 4
math: true
---

Actor-Critic (AC) methods are widely-used policy gradient that employ critics to facilitate training.
AC methods can achieve lower variance and better sample efficiency than REINFORCE, but this requires careful design and tuning of the critic to ensure stable training.
In Multi-Agent Reinforcement Learning (MARL), Actor-Critic methods can be instantiated as Multi-Agent Actor-Critic (MAAC) and Independent Actor-Critic (IAC).

## MAAC

Multi-Agent Actor-Critic (MAAC) uses a Centralized Critic (CC) across agents to evaluate the values of joint histories {{< katex >}}V_{\boldsymbol{\phi}}(\mathbf{h}_t){{< /katex >}} or joint history-action pairs {{< katex >}}Q_{\boldsymbol{\psi}}(\mathbf{h}_t, \mathbf{a}_t){{< /katex >}}.
The policy gradient of each agent is:

{{< katex display=true >}}
\nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\boldsymbol{\pi}}\left[\sum_{t=0}^{H-1} \nabla_{\theta_i} \log \pi_{\theta_i}(a_{i,t}|h_{i,t})\,\boldsymbol{\delta}_t\right]
{{< /katex >}}

where {{< katex inline=true >}}\boldsymbol{\delta}_t = r_t + \gamma V_{\boldsymbol{\phi}}(\mathbf{h}_{t+1}) - V_{\boldsymbol{\phi}}(\mathbf{h}_{t}){{< /katex >}}, and the critic is updated by:

{{< katex display=true >}}
\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{\boldsymbol{\pi}}\left[\sum_{t=0}^{H-1} \big(r_t + \gamma V_{\phi}(\mathbf{h}_{t+1}) - V_{\phi}(\mathbf{h}_{t})\big)^2\right].
{{< /katex >}}

{{% hint info %}}
**MAACConfig** parameters:
- `num_agents`: Number of agents
- `num_turns`: Number of turns
- `critic_type`: Critic target type (`v` for V(h), `q` for Q(h,a))
- `num_train_epochs`: Number of training epochs
- `agent_learning_rate`: Learning rate for agents
- `critic_learning_rate`: Learning rate for shared critic
- `value_loss_coef`: Weight on critic loss
- `advantage_normalization`: Whether to normalize advantages before updates
- `rollout_buffer_size`: Number of samples to collect per agent before an update
- `train_batch_size`: Mini-batch size within each update
- `max_new_tokens`: Maximum tokens to generate per completion
- `temperature`: Temperature for sampling
- `top_p`: Top-p for nucleus sampling
- `top_k`: Top-k for sampling
- `num_generations`: Number of generations per prompt per agent
- `external_prompt_passthrough`: Use external prompts directly in multi-turn
- `discount`: Discount factor for multi-turn returns
- `early_termination_threshold`: Optional early-stop threshold for multi-turn
- `eval_interval`: Evaluation interval (in training batches)
- `eval_num_samples`: Number of evaluation samples per interval
- `eval_batch_size`: Eval dataloader batch size
- `logging_steps`: Log every N training batches
{{% /hint %}}

{{% hint info %}}
**MAACTrainer** setup:

- `agent_model` or `agents`: Actor model identifier string for homogeneous agents, or list of agent models (multi-agent `agent_model` must be a string)
- `critic_model` or `critics`: Required single shared critic (either one identifier or a 1-element list)
- `tokenizer`: Tokenizer (required)
- `reward_func`: Callable returning rewards (required)
- `reward_processor`: Optional reward post-processor
- `formatters`: Single callable or list for per-agent prompt formatting
- `args`: Instance of `MAACConfig` (optional)
- `train_dataset`: Training dataset (required)
- `eval_dataset`: Optional evaluation dataset
- `model_config`: Extra model kwargs (optional)
- `wandb_config`: Weights & Biases logging config (optional)
- `metrics_callback`: Optional callback for custom metrics
{{% /hint %}}

{{% hint warning %}}
For simplicity, IAC computes the policy gradient using the current policy's samples without importance sampling or ratio clipping. The `value_clip_range` is not applicable in MAAC.
{{% /hint %}}

{{% hint warning %}}
The trainer uses a fixed training DataLoader batch size of 1. For `num_turns > 1`, provide an `external_transition` and set `num_generations=1`. The training use batch gradient descent by default, where `train_batch_size`=`rollout_buffer_size`.
{{% /hint %}}

## IAC

Independent Actor-Critic (IAC) optimizes each agent's policy independently while using joint returns from multiple agents. Each agent maintains its own actor and critic, other agents serve as part of the environment. The policy gradient for each agent is:

{{< katex display=true >}}
\nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\boldsymbol{\pi}}\left[\sum_{t=0}^{H-1} \nabla_{\theta_i} \log \pi_{\theta_i}(a_{i,t}|h_{i,t})\,\delta_{i,t}\right]
{{< /katex >}}

where {{< katex inline=true >}}\delta_{i,t} = r_t + \gamma V_{\phi_i}(h_{i,t+1}) - V_{\phi_i}(h_{i,t}){{< /katex >}}.


CoMLRL supports two IAC variants:

- **Separate Critic**: Uses an independent model for value estimation separate from the actor. It provides more stable training but occupies larger storage and VRAM usage.

- **Shared Model**: Attaches a small value-head to the transformer backbone, sharing the actor model's history (or history-action) representations to reduce the space costs.

The critics are updated by minimizing the TD error:

{{< katex display=true >}}
\mathcal{L}(\phi_i) = \mathbb{E}_{\boldsymbol{\pi}}\left[\sum_{t=0}^{H-1} \big(r_t + \gamma V_{\phi_i}(h_{i,t+1}) - V_{\phi_i}(h_{i,t})\big)^2\right].
{{< /katex >}}

When using shared model `use_separate_critic=false`, a value clip `value_clip_range` can be applied to improve training stability.

{{< katex display=true >}}
L(\phi_i) = \max\Big( (V_{\phi_i}(h_t) - \hat{V}_t)^2,\ (V_{\phi_i}^{\text{clip}}(h_t) - \hat{V}_t)^2 \Big)
\\ V_{\phi_i}^{\text{clip}}(h_t) = V_{\phi_i}^{\text{old}}(h_t) + \mathrm{clip}(V_{\phi_i}(h_t) - V_{\phi_i}^{\text{old}}(h_t), -\epsilon, \epsilon),
{{< /katex >}}

{{% hint info %}}
**IACConfig** provides parameters for configuring Independent Actor-Critic training:

- `num_agents`: Number of agents
- `num_turns`: Number of turns
- `num_train_epochs`: Number of training epochs
- `agent_learning_rate`: Learning rate for agents
- `critic_learning_rate`: Learning rate for critic
- `value_loss_coef`: Coefficient for value loss
- `value_clip_range`: Clipping range for value function
- `advantage_normalization`: Whether to normalize advantages
- `rollout_buffer_size`: Number of samples to collect before update
- `train_batch_size`: Mini-batch size for policy updates
- `max_new_tokens`: Maximum new tokens to generate
- `temperature`: Temperature for sampling
- `top_p`: Top-p for nucleus sampling
- `top_k`: Top-k for sampling
- `num_generations`: Number of generations per prompt per agent
- `use_separate_critic`: Whether to use separate critic model
- `critic_type`: Critic target type (`v` for V(h), `q` for Q(h,a))
- `critic_value_head_hidden_dim`: Hidden dimension for critic value head
- `value_head_hidden_dim`: Hidden dimension for value head in shared-critic mode
- `external_prompt_passthrough`: Use external prompts directly in multi-turn
- `discount`: Discount factor for multi-turn returns
- `early_termination_threshold`: Optional early-stop threshold for multi-turn
- `eval_interval`: Evaluation interval (in training batches)
- `eval_num_samples`: Number of evaluation samples per interval
- `eval_batch_size`: Eval dataloader batch size
- `logging_steps`: Log every N training batches
{{% /hint %}}

{{% hint info %}}
**IACTrainer** trains agents using Independent Actor-Critic:

- `agent_model` or `agents`: Model identifier string for homogeneous agents, or list of agent models (multi-agent `agent_model` must be a string)
- `critic_model` or `critics`: Critic identifier or list of critic models when `use_separate_critic=true`
- `tokenizer`: The tokenizer (required)
- `reward_func`: Callable that returns a list of floats (required)
- `reward_processor`: Optional processor to apply to rewards
- `formatters`: Single callable or list of callables for each agent to format dataset items into prompts
- `args`: Instance of `IACConfig` (optional)
- `train_dataset`: Training dataset (required)
- `eval_dataset`: Evaluation dataset (optional)
- `model_config`: Model configuration dict (optional)
- `wandb_config`: Configuration for Weights & Biases logging (optional)
- `metrics_callback`: Optional callback for custom metrics
- `external_transition`: Optional transition function required for multi-turn training
{{% /hint %}}

{{% hint warning %}}
For simplicity, IAC computes the policy gradient using the current policy's samples without importance sampling or ratio clipping. In shared-critic mode (`use_separate_critic=false`), value heads are attached to the actor models (do not pass `critic_model`/`critics`; passing them raises an error), and agents may be homogeneous or heterogeneous; this mode can be less stable, and `value_clip_range` only applies there. In separate-critic mode (`use_separate_critic=true`), pass a `critics` list with length equal to `num_agents` or a single `critic_model` to be broadcast; critic models may differ from actor models.
{{% /hint %}}

{{% hint warning %}}
The trainer uses a fixed training DataLoader batch size of 1. For `num_turns > 1`, provide an `external_transition` and set `num_generations=1`. The training use batch gradient descent by default, where `train_batch_size`=`rollout_buffer_size`.
{{% /hint %}}
