# Hyperparameter List

**CoMLRL (source of truth)**

**IAC (Independent Actor-Critic)**
- `actor_learning_rate`
- `critic_learning_rate`
- `weight_decay`
- `adam_beta1`
- `adam_beta2`
- `adam_epsilon`
- `max_grad_norm`
- `rollout_buffer_size`
- `train_batch_size` (defaults to `rollout_buffer_size`)
- `value_clip_range`
- `value_loss_coef`
- `advantage_normalization`
- `max_new_tokens`
- `temperature`
- `top_p`
- `top_k`
- `do_sample`
- `num_train_epochs`
- `use_separate_critic`
- `critic_model_name_or_path`
- `critic_type` (`v` or `q`)
- `critic_value_head_hidden_dim`
- `value_head_hidden_dim`
- `pad_token_id`
- `num_agents`
- `num_turns`
- `external_prompt_passthrough`
- `discount`
- `num_generations`
- `eval_interval`
- `eval_num_samples`
- `eval_batch_size`
- `early_termination_threshold`
- `logging_steps`

**MAAC (Multi-Agent Actor-Critic)**
- `actor_learning_rate`
- `critic_learning_rate`
- `weight_decay`
- `adam_beta1`
- `adam_beta2`
- `adam_epsilon`
- `max_grad_norm`
- `rollout_buffer_size`
- `train_batch_size` (defaults to `rollout_buffer_size`)
- `value_loss_coef`
- `advantage_normalization`
- `max_new_tokens`
- `temperature`
- `top_p`
- `top_k`
- `do_sample`
- `num_train_epochs`
- `pad_token_id`
- `num_agents`
- `num_generations`
- `critic_model_name_or_path`
- `num_turns`
- `external_prompt_passthrough`
- `discount`
- `critic_type` (`v` or `q`)
- `early_termination_threshold`
- `eval_interval`
- `eval_num_samples`
- `eval_batch_size`
- `logging_steps`

**MAGRPO (Multi-Agent Group-Relative Policy Optimization)**
- `num_train_epochs`
- `learning_rate`
- `weight_decay`
- `logging_steps`
- `num_agents`
- `num_generations`
- `max_new_tokens`
- `temperature`
- `top_p`
- `top_k`
- `num_turns`
- `discount`
- `joint_mode` (`aligned` or `cross`)
- `termination_threshold`
- `external_prompt_passthrough`
- `eval_interval`
- `eval_num_samples`
- `eval_batch_size`
- `rollout_buffer_size`
- `advantage_mode` (`mean`, `max`, `rloo`, `raw`)
- `dataloader_drop_last`
- `dataloader_num_workers`

**MAREINFORCE / MAREMAX / MARLOO**
Same as MAGRPO, but `advantage_mode` is fixed.
- `advantage_mode=raw` (MAREINFORCE)
- `advantage_mode=max` (MAREMAX)
- `advantage_mode=rloo` (MARLOO)

**Downstream Extras (not in CoMLRL configs)**
- `LLM_Collab_Writing`: `critic_model` (alias of `critic_model_name_or_path`), `critic_model_kwargs`
- `LLM_Collab_Code_Generation`: `critic_model` (alias), `critic_model_kwargs`, `reward_shift`
- `LLM_Collab_Code_Completion`: `critic_model` (alias), `critic_model_kwargs`
- `LLM_Collab_MC`: `critic_model` (alias), `critic_model_kwargs`
