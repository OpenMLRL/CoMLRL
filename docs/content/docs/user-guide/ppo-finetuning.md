---
title: Multi-Agent PPO
weight: 3
math: true
---

PPO is a widely used policy gradient method that employs generalized advantage estimation to estimate advantages, reducing the high variance and long rollout times in Monte Carlo methods, e.g., REINFORCE. PPO has also been used for LLM fine-tuning, e.g., [trl](https://huggingface.co/docs/trl/main/en/ppo_trainer), [verl](https://verl.readthedocs.io/en/latest/algo/ppo.html), [LLaMA Factory](https://llamafactory.readthedocs.io/en/latest/advanced/trainers.html#ppo).

## IPPO

Independent PPO ([IPPO](https://arxiv.org/abs/2011.09533)) optimizes each agent's policy independently while using joint returns from multiple agents. Each agent maintains its own actor and critic, other agents serve as part of the environment. The policy objective is:

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{o_{i,0} \sim \mathcal{D}, h_i \sim \pi_{\theta_i}}\left[\log \pi_{\theta_i}(a_{i,t}|h_{i,t}) \cdot \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{i,t+l} + \beta \mathcal{H}(\pi_{\theta_i})\right]
{{< /katex >}}

where {{< katex inline=true >}}\delta_{i,t} = r_{i,t} + \gamma V_\phi(h_{i,t+1}) - V_\phi(h_{i,t}){{< /katex >}} is the temporal difference error, {{< katex inline=true >}}\gamma{{< /katex >}} is the discount factor, {{< katex inline=true >}}\lambda{{< /katex >}} is the GAE parameter that balances bias and variance, and {{< katex inline=true >}}\mathcal{H}(\pi_{\theta_i}){{< /katex >}} is the entropy bonus with coefficient {{< katex inline=true >}}\beta{{< /katex >}}.

CoMLRL supports two IPPO architectures for critic implementation:

- **Separate Critic**: Uses an independent model dedicated to value estimation, completely separate from the actor. It provides more stable training but requires longer training time and larger VRAM usage.

- **Value Head**: Attaches a small value prediction head directly to the actor model, sharing the base model's representations. It reduces VRAM usage, but since both actor and critic share the same model, gradient errors can be amplified during training.

{{% hint info %}}
**IPPOConfig** provides parameters for configuring the PPO training:

- `output_dir`: Directory to save outputs
- `actor_learning_rate`: Learning rate for actor
- `critic_learning_rate`: Learning rate for critic
- `weight_decay`: Weight decay for AdamW optimizer
- `adam_beta1`, `adam_beta2`, `adam_epsilon`: Adam optimizer parameters
- `max_grad_norm`: Maximum gradient norm for clipping
- `rollout_buffer_size`: Number of samples to collect before update
- `mini_batch_size`: Mini-batch size for PPO updates
- `ppo_epochs`: Number of optimization epochs per rollout
- `value_clip_range`: Clipping range for value function
- `value_loss_coef`: Coefficient for value loss
- `entropy_coef`: Coefficient for entropy bonus
- `advantage_normalization`: Whether to normalize advantages
- `max_new_tokens`: Maximum new tokens to generate
- `temperature`: Temperature for sampling
- `top_p`: Top-p for nucleus sampling
- `top_k`: Top-k for sampling
- `do_sample`: Whether to use sampling
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device, must be 1
- `use_separate_critic`: Whether to use separate critic model
- `critic_model_name_or_path`: Model identifier for separate critic
- `critic_value_head_hidden_dim`: Hidden dimension for critic value head
- `value_head_hidden_dim`: Hidden dimension for actor value head
- `num_agents`: Number of agents
- `num_turns`: Number of turns, currently only supports 1
- `reward_norm_eps`: Epsilon for reward normalization
{{% /hint %}}

{{% hint info %}}
**IPPOTrainer** trains agents using Independent PPO:

- `model`: Model string or PreTrainedModel instance (required for single-agent, must be string for multi-agent)
- `tokenizer`: The tokenizer (required)
- `reward_func`: Callable that returns a list of floats (required)
- `reward_processor`: Optional processor to apply to rewards
- `formatters`: Single callable or list of callables for each agent to format dataset items into prompts
- `args`: Instance of `IPPOConfig` (optional)
- `train_dataset`: Training dataset (required)
- `eval_dataset`: Evaluation dataset (optional)
- `model_config`: Model configuration dict (optional)
- `wandb_config`: Configuration for Weights & Biases logging (optional)
- `metrics_callback`: Optional callback for custom metrics
{{% /hint %}}

{{% hint warning %}}
CoMLRL implements on-policy IPPO, which computes the policy gradient using the current policy's samples without importance sampling or ratio clipping.
{{% /hint %}}

{{% hint warning %}}
The trainer enforces `per_device_train_batch_size=1` and currently only supports single-turn training (`num_turns=1`).
{{% /hint %}}
