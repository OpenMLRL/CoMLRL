---
title: MAPL
weight: 3
aliases:
  - /docs/user-guide/multi-agent-preference-learning/
math: true
---

Multi-Agent Preference Learning (MAPL) improves collaborating agents by comparing joint responses to the same task. A preference identifies a better and a worse joint response, so each agent learns in the context of the team's outcome.

CoMLRL provides two approaches: **MADPO** directly optimizes the policies from preference pairs, while **MARLHF** first learns a joint reward model and then trains the policies with that reward. Their iterative variants refresh the preference data as training progresses. MAPL names this family of methods; the public trainer classes are `MADPOTrainer`, `MARLHFTrainer`, `MADPOIterTrainer`, and `MARLHFIterTrainer`.

Explore the three parts of a preference-training setup:

- [Training Paradigms]({{< relref "/docs/mapl/training-paradigms" >}}): Actor and comparator generation modes, joint-response adapters, and iterative training.
- [Comparator]({{< relref "/docs/mapl/comparator" >}}): Current policies, model checkpoints, historical policies, and remote APIs.
- [Replay Buffer]({{< relref "/docs/mapl/replay-buffer" >}}): Current-round data, recent iterations, full history, and age-weighted sampling.

## Preference Pairs

For a task with per-agent prompts {{< katex >}}\mathbf{h}=(h_1,\ldots,h_N){{< /katex >}}, the trainer samples candidate joint responses {{< katex >}}\mathbf{a}^{k}=(a_1^k,\ldots,a_N^k){{< /katex >}}. Candidates with the same index are aligned across agents and receive a shared scalar task reward. A pair is ordered as {{< katex >}}\mathbf{a}^{+}\succ\mathbf{a}^{-}{{< /katex >}} when the first response has a strictly higher reward. Ties are discarded.

The training dataset contains task items used to construct prompts. By default, the formatter reads each item's `prompt` field; custom `formatters` can use other fields or provide different prompts for each role. The trainers generate and score their own preference pairs from these items. The base MADPO and MARLHF trainers collect those pairs once before optimization.

{{% hint warning %}}
All four preference trainers currently require `num_turns=1` and aligned joint responses (`joint_mode="aligned"`). The iterative variants repeat data collection and training; an iteration is not a dialog turn.
{{% /hint %}}

## MADPO

Multi-Agent Direct Preference Optimization (MADPO) uses a joint preference loss across the agents. Let {{< katex >}}s_{\theta_i}(h_i,a_i){{< /katex >}} denote the completion log-probability score computed by agent {{< katex >}}i{{< /katex >}}. Its preference margin is:

{{< katex display=true >}}
\Delta_i = s_{\theta_i}(h_i,a_i^{+})-s_{\theta_i}(h_i,a_i^{-}).
{{< /katex >}}

The implemented loss is:

{{< katex display=true >}}
\mathcal{L}_{\mathrm{MADPO}}(\boldsymbol{\theta})
=-\mathbb{E}_{(\mathbf{h},\mathbf{a}^{+},\mathbf{a}^{-})\sim\mathcal{D}_{\mathrm{pref}}}
\left[\log\sigma\!\left(\beta\sum_{i=1}^{N}\Delta_i\right)\right].
{{< /katex >}}

Here {{< katex >}}\sigma{{< /katex >}} is the sigmoid function and {{< katex >}}\beta{{< /katex >}} is `dpo_beta`. During each agent's update, the other agents' margins are detached from the computation graph. The preference signal therefore depends on the joint response, while gradients flow through the agent being updated.

{{% hint info %}}
**MADPOConfig** parameters:

- `num_agents`: Number of agents in decentralized mode, or task roles in centralized mode
- `collaboration_mode`: `"decentralized"` (default) or `"centralized"`
- `num_train_epochs`: Number of optimization epochs over the collected preference pairs
- `agent_learning_rate`: Learning rate for the policy optimizers
- `preference_num_candidates`: Candidates sampled per task and actor; must be at least 2
- `preference_pairs_per_sample`: Maximum selected pairs per task, or `None` for no limit
- `pair_selection`: `"reward_gap"` selects the largest reward gaps, `"random"` shuffles valid pairs, and `"all"` retains every non-tied pair
- `dpo_beta`: Positive multiplier on the joint preference margin
- `train_batch_size`: Number of preference pairs per update; defaults to `rollout_buffer_size`
- `max_new_tokens`, `temperature`, `top_p`, `top_k`: Candidate generation settings
- `parallel_training`, `agent_devices`: Policy execution mode and device placement
- `eval_interval`, `eval_num_samples`, `eval_batch_size`: Evaluation schedule and sample sizes
- `logging_steps`: Training logging interval
{{% /hint %}}

{{% hint info %}}
**MADPOTrainer** setup:

- `agent_model` or `agents`: Model identifier for homogeneous agents, or a list of model identifiers/instances
- `num_agents`: Number of agents or task roles; match `args.num_agents`
- `tokenizer`: Tokenizer, or one tokenizer per actor
- `train_dataset`: Task items from which preference candidates are generated
- `eval_dataset`: Optional evaluation dataset
- `reward_func`: Required callable returning one shared reward per aligned joint response
- `reward_processor`: Optional post-processing of task rewards
- `formatters`: One prompt formatter shared across roles, or a list of role-specific formatters
- `args`: `MADPOConfig` instance
- `model_config`: Optional model loading configuration
- `wandb_config`: Optional Weights & Biases configuration
- `eval_logger`, `eval_aggregator`: Optional evaluation metrics hooks, as in `MAGRPOTrainer`
- `centralized_comparator_adapter`: Prompt builder and response parser for centralized actors or comparators
{{% /hint %}}

{{% hint warning %}}
MADPO currently uses a reference-free preference objective: its loss contains no reference-policy log-probability ratio and does not apply the optional reference KL settings inherited from the shared configuration. Set `preference_pairs_per_sample=None` when using `pair_selection="all"`. If every candidate reward is tied, the base trainer has no preference pairs to optimize and returns without updates.

The current decentralized scoring helper omits the last completion token after trimming padding; the centralized helper scores the full completion. The formula above uses the score returned by the implementation. This token-boundary difference is an implementation limitation.
{{% /hint %}}

## MARLHF

Multi-Agent Reinforcement Learning from Human Feedback (MARLHF) separates reward learning from policy learning. In the current implementation, preference labels are obtained from the supplied task `reward_func`; the public training entry point starts from task items and generates the comparisons.

The `JointRewardModel` reads the combined prompts and responses of the team. It uses a causal language model backbone and a scalar reward head applied to the final non-padding token. Given a preferred and a rejected response, the reward model minimizes:

{{< katex display=true >}}
\mathcal{L}_{\mathrm{reward}}(\phi)
=-\mathbb{E}_{\mathcal{D}_{\mathrm{pref}}}
\left[\log\sigma\!\left(r_{\phi}(\mathbf{h},\mathbf{a}^{+})-r_{\phi}(\mathbf{h},\mathbf{a}^{-})\right)\right].
{{< /katex >}}

After reward training, its scores are used without reward-model gradients during online policy optimization. Choose the policy optimizer with `rl_algorithm`: `"magrpo"`, `"mareinforce"`, `"maremax"`, `"marloo"`, `"maac"`, or `"iac"`. Evaluation continues to use the original task reward, allowing policy performance to be checked independently of the learned reward.

{{% hint info %}}
**MARLHFConfig** parameters:

- Shared MADPO settings control preference collection, collaboration, generation, and policy training; `dpo_beta` is not used by the reward-model loss
- `rl_algorithm`: Online policy-training algorithm; defaults to `"magrpo"`
- `num_generations`: Online candidates per prompt and actor; at least 2 for the MAGRPO family, or at least 1 for MAAC/IAC
- `reward_model_name`: Model identifier for the reward backbone; defaults to the actor model identifier when available
- `reward_model_device`: Reward-model device; defaults to `"cuda:0"`, or use `None` to inherit the trainer device
- `reward_learning_rate`, `reward_num_train_epochs`, `reward_train_batch_size`: Reward-model optimization settings
- `reward_max_length`: Optional token limit for the combined reward-model input
- `reward_freeze_backbone`: Train only the scalar reward head when `True`
- `critic_model_name`, `critic_learning_rate`, `critic_devices`: Critic settings for an actor-critic backend
- `critic_type`: `"v"` or `"q"` for an actor-critic backend
- `use_separate_critic`: Select separate critics or actor-attached value heads for IAC
- `value_loss_coef`, `value_clip_range`: Critic loss weight and IAC shared-value-head clipping; see [Actor-Critic]({{< relref "/docs/marl/multi-agent-actor-critic" >}})
- `reference_kl_enabled`, `reference_kl_coef`, `reference_devices`: Optional frozen-reference regularization for the online RL stage
{{% /hint %}}

{{% hint info %}}
**MARLHFTrainer** setup:

- Uses the same task dataset, actor, tokenizer, reward, and formatter inputs as `MADPOTrainer`, with `args=MARLHFConfig(...)`
- `critic_model` or `critics`: Optional critic sources for the MAAC/IAC backend. `critic_model` must be a model identifier string; loaded model instances must be supplied through `critics`. Use a one-element list for MAAC or one entry per actor for separate-critic IAC. Without explicit sources, `critic_model_name` or the actor model identifier is used
- `metrics_callback`: Optional custom metrics callback for the actor-critic backend
{{% /hint %}}

{{% hint warning %}}
Provide `reward_model_name` when actors are supplied as model objects without a model identifier. Likewise, MAAC or separate-critic IAC needs explicit critic sources or a usable critic model identifier. MARLHF remains single-turn even when the selected RL backend supports multi-turn training on its own.
{{% /hint %}}

## Iterative Training

<span id="api-madpoiterconfig"></span>
<span id="api-marlhfiterconfig"></span>

`MADPOIterTrainer` and `MARLHFIterTrainer` alternate preference collection with optimization. See [Training Paradigms]({{< relref "/docs/mapl/training-paradigms#iterative-training" >}}) for the training cycle and configuration, [Comparator]({{< relref "/docs/mapl/comparator" >}}) for comparison policies, and [Replay Buffer]({{< relref "/docs/mapl/replay-buffer" >}}) for historical data selection.

## Collaboration Modes

Preference training supports separate policies per role or one actor generating the complete joint response. See [Training Paradigms]({{< relref "/docs/mapl/training-paradigms#collaboration-modes" >}}) for the supported actor/comparator layouts and centralized adapters.
