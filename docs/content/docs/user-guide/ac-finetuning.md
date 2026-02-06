---
title: Multi-Agent Actor-Critic
weight: 3
math: true
---

Actor-Critic methods are widely used policy gradient approaches that employ critics to estimate advantages, reducing the high variance and supporting online training. Many LLM fine-tuning frameworks implement actor-critic training (e.g., [trl](https://huggingface.co/docs/trl), [verl](https://verl.readthedocs.io/en/latest/), [LLaMA Factory](https://llamafactory.readthedocs.io/en/latest/advanced/trainers.html)).

## IAC

Independent Actor-Critic (IAC) optimizes each agent's policy independently while using joint returns from multiple agents. Each agent maintains its own actor and critic, other agents serve as part of the environment. The policy objective is:

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{o_{i,0} \sim \mathcal{D}, h_i \sim \pi_{\theta_i}}\left[\log \pi_{\theta_i}(a_{i,t}|h_{i,t}) \cdot \delta_{i,t}\right]
{{< /katex >}}

where {{< katex inline=true >}}\delta_{i,t} = r_{i,t} + \gamma V_{\phi_i}(h_{i,t+1}) - V_{\phi_i}(h_{i,t}){{< /katex >}} is the (single-step) temporal difference error and {{< katex inline=true >}}\gamma{{< /katex >}} is the discount factor. Use `critic_type='q'` to switch to a Q-value critic {{< katex inline=true >}}Q(h_t, a_t){{< /katex >}}; the default is `critic_type='v'`.

CoMLRL supports two IAC architectures for critic implementation:

- **Separate Critic**: Uses an independent model dedicated to value estimation, completely separate from the actor. It provides more stable training but requires longer training time and larger VRAM usage.

- **Shared Model**: Attaches a small value prediction head directly to the transformer backbone, sharing the actor model's representations to reduce the time and space costs.

{{% hint info %}}
**IACConfig** provides parameters for configuring Independent Actor-Critic training:

- `num_agents`: Number of agents
- `num_turns`: Number of turns
- `critic_model_name_or_path`: Model identifier for separate critic
- `num_train_epochs`: Number of training epochs
- `actor_learning_rate`: Learning rate for actor
- `critic_learning_rate`: Learning rate for critic
- `value_loss_coef`: Coefficient for value loss
- `value_clip_range`: Clipping range for value function
- `rollout_buffer_size`: Number of samples to collect before update
- `train_batch_size`: Mini-batch size for policy updates
- `max_new_tokens`: Maximum new tokens to generate
- `temperature`: Temperature for sampling
- `top_p`: Top-p for nucleus sampling
- `top_k`: Top-k for sampling
- `use_separate_critic`: Whether to use separate critic model
- `discount`: Discount factor for multi-turn returns
- `early_termination_threshold`: Optional early-stop threshold for multi-turn
- `eval_interval`: Evaluation interval (in training batches)
- `eval_num_samples`: Number of evaluation samples per interval
- `eval_batch_size`: Eval dataloader batch size
- `logging_steps`: Log every N training batches
- `advantage_normalization`: Whether to normalize advantages
- `do_sample`: Whether to use sampling
- `critic_type`: Critic target type (`v` for V(h), `q` for Q(h,a))
- `critic_value_head_hidden_dim`: Hidden dimension for critic value head
- `value_head_hidden_dim`: Hidden dimension for actor value head
- `pad_token_id`: Padding token id
- `num_generations`: Number of generations per prompt per agent
- `external_prompt_passthrough`: Use external prompts directly in multi-turn
{{% /hint %}}

{{% hint info %}}
**IACTrainer** trains agents using Independent Actor-Critic:

- `model`: Model string or PreTrainedModel instance (required for single-agent, must be string for multi-agent)
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
For simplicity, IAC computes the policy gradient using the current policy's samples without importance sampling or ratio clipping.
{{% /hint %}}

{{% hint warning %}}
The trainer uses a fixed training DataLoader batch size of 1. For `num_turns > 1`, provide an `external_transition` and set `num_generations=1`.
{{% /hint %}}

## MAAC

Multi-Agent Actor-Critic (MAAC) shares a centralized critic across agents. The policy objective mirrors IAC with a joint value baseline:

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{h_t \sim \mathcal{D},\, a_t \sim \pi_{\theta}}\left[\log \pi_{\theta_i}(a_{i,t}|h_{i,t}) \cdot \mathbf{\delta}_t\right]
{{< /katex >}}

where {{< katex inline=true >}}\mathbf{\delta}_t = r_t + \gamma V_{\phi}(\mathbf{h}_{t+1}) - V_{\phi}(\mathbf{h}_{t}){{< /katex >}} uses the shared critic on the joint prompt/history. Set `critic_type='q'` to condition the critic on joint responses via {{< katex inline=true >}}Q(\mathbf{h}_t, \mathbf{a}_t){{< /katex >}}.

{{% hint info %}}
**MAACConfig** parameters:
- `num_agents`: Number of actors
- `num_turns`: Number of turns
- `critic_model_name_or_path`: Required identifier for the shared critic
- `critic_type`: Critic target type (`v` for V(h), `q` for Q(h,a))
- `num_train_epochs`: Number of training epochs
- `actor_learning_rate`: Learning rate for actors
- `critic_learning_rate`: Learning rate for shared critic
- `value_loss_coef`: Weight on critic loss
- `rollout_buffer_size`: Number of samples to collect per agent before an update
- `train_batch_size`: Mini-batch size within each update
- `max_new_tokens`: Maximum tokens to generate per completion
- `temperature`: Temperature for sampling
- `top_p`: Top-p for nucleus sampling
- `top_k`: Top-k for sampling
- `discount`: Discount factor for multi-turn returns
- `early_termination_threshold`: Optional early-stop threshold for multi-turn
- `eval_interval`: Evaluation interval (in training batches)
- `eval_num_samples`: Number of evaluation samples per interval
- `eval_batch_size`: Eval dataloader batch size
- `logging_steps`: Log every N training batches
- `advantage_normalization`: Whether to normalize advantages before updates
- `do_sample`: Whether to use sampling
- `pad_token_id`: Padding token id
- `num_generations`: Number of generations per prompt per agent
- `external_prompt_passthrough`: Use external prompts directly in multi-turn
{{% /hint %}}

{{% hint info %}}
**MAACTrainer** setup:

- `model`: Actor model identifier/string (required)
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
