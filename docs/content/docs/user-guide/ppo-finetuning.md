---
title: Multi-Agent PPO
weight: 3
math: true
---

PPO is a widely used policy gradient method that employs generalized advantage estimation to estimate advantages, reducing the high variance and long rollout times in Monte Carlo methods, e.g., REINFORCE. PPO has also been used for LLM fine-tuning, e.g., [trl](https://huggingface.co/docs/trl/main/en/ppo_trainer), [verl](https://verl.readthedocs.io/en/latest/algo/ppo.html), [LLaMA Factory](https://llamafactory.readthedocs.io/en/latest/advanced/trainers.html#ppo).

## IPPO

Independent PPO ([IPPO](https://arxiv.org/abs/2011.09533)) optimizes each agent's policy independently while using joint returns from multiple agents. Each agent maintains its own actor and critic, other agents serve as part of the environment.

{{< katex display=true >}}
L^{\text{CLIP}}(\theta_i) = \mathbb{E}\left[\min\left(r_i(\theta_i) \hat{A}_i, \text{clip}(r_i(\theta_i), 1-\epsilon, 1+\epsilon) \hat{A}_i\right)\right]
{{< /katex >}}

where {{< katex inline=true >}}r_i(\theta_i) = \frac{\pi_{\theta_i}(a_i|s)}{\pi_{\theta_i^{\text{old}}}(a_i|s)}{{< /katex >}} is the probability ratio, {{< katex inline=true >}}\hat{A}_i{{< /katex >}} is the advantage estimate, and {{< katex inline=true >}}\epsilon{{< /katex >}} is the clipping parameter.

**Separate Critic**: This architecture uses an independent model dedicated to value estimation, completely separate from the actor. It provides better separation of concerns between policy and value learning, allows different learning rates for actor and critic, and enables independent optimization schedules. However, it requires significantly more memory as it maintains two full models, and increases computational cost during training.

**Value Head**: This architecture attaches a small value prediction head directly to the actor model, sharing the base model's representations. It is more memory-efficient since only one base model is loaded, simpler to implement and maintain, and naturally aligns actor and critic representations. However, it tightly couples actor and critic updates, uses the same learning rate for both (unless explicitly separated), and may lead to interference between policy and value learning.

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

The trainer enforces `per_device_train_batch_size=1` and currently only supports single-turn training (`num_turns=1`).
{{% /hint %}}
