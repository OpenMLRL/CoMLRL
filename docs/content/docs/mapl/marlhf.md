---
title: Multi-Agent RLHF
weight: 2
math: true
---

Multi-Agent Reinforcement Learning from Human Feedback (MARLHF) fits a joint reward model to preference pairs, then trains the policies with RL on the learned reward. CoMLRL generates the preference labels with the task `reward_func`.

The training flow is: **task items → preference pairs → Stage 1: joint reward learning → Stage 2: policy reinforcement learning**. Because the reward model scores the team's complete response, separate actors learn from one shared reward. [MADPO]({{< relref "/docs/mapl/madpo" >}}) instead optimizes the policies directly on the pairs, without a reward model.

In the diagram, both agents' responses are scored together and ranked into pairs; Stage 1 fits the reward model on these pairs, and Stage 2 trains each policy on advantages from the frozen model.

{{< rollout-diagram diagram="marlhf-training" title="MARLHF training flow" description="Preference pairs fit a joint reward model, whose frozen scores then train both policies with RL." >}}

## Preference Pairs

Each agent {{< katex >}}i\in\mathcal{I}=\{1,\ldots,n\}{{< /katex >}} samples {{< katex >}}a_{i,t}\sim\pi_{\theta_i}(\cdot\mid h_{i,t}){{< /katex >}} from its local history; the responses form the joint action {{< katex >}}\mathbf{a}_t=\{a_{1,t},\ldots,a_{n,t}\}{{< /katex >}} with joint history {{< katex >}}\mathbf{h}_t{{< /katex >}}. Only a single turn is currently supported (horizon {{< katex >}}H=1{{< /katex >}}), so {{< katex >}}h_{i,0}{{< /katex >}} is agent {{< katex >}}i{{< /katex >}}'s prompt and {{< katex >}}\mathbf{a}_0{{< /katex >}} is the joint response.

For each task item in `train_dataset`, every agent samples `preference_num_candidates` responses from its formatted prompt. Responses with the same candidate index form a joint response and share one reward from `reward_func` (after the optional `reward_processor`). Of two joint responses with different rewards, the higher one is the winner {{< katex >}}w{{< /katex >}} and the other the loser {{< katex >}}l{{< /katex >}}; ties are discarded. Each pair {{< katex >}}(\mathbf{h}_H^{w},\mathbf{h}_H^{l}){{< /katex >}} holds the two joint histories of one task.

The base trainer collects the pairs once, before either stage; `preference_pairs_per_sample` and `pair_selection` choose which non-tied pairs each task keeps. If no task yields a non-tied pair, training returns without fitting a reward model or updating the policies.

{{% hint warning %}}
MARLHF currently requires `num_turns=1` and `joint_mode="aligned"`, even when the selected RL backend supports multi-turn training.
{{% /hint %}}

## Stage 1: Joint Reward Learning

The `JointRewardModel` {{< katex >}}\widetilde{R}_{\psi}(\mathbf{h}_t,\mathbf{a}_t){{< /katex >}} concatenates all agents' prompts and responses into one text and applies a scalar head to the causal-LM backbone's last hidden state at the final non-padding token. Reward learning minimizes the Bradley–Terry loss over the training pairs {{< katex >}}\mathcal{B}{{< /katex >}}:

{{< katex display=true >}}
\mathcal{L}_{\mathrm{RM}}(\psi)
=-\mathbb{E}_{(\mathbf{h}_H^{w},\mathbf{h}_H^{l})\sim\mathcal{B}}
\left[\log\sigma\left(\sum_{t=0}^{H-1}\Big[\widetilde{R}_{\psi}(\mathbf{h}_t^{w},\mathbf{a}_t^{w})-\widetilde{R}_{\psi}(\mathbf{h}_t^{l},\mathbf{a}_t^{l})\Big]\right)\right],
{{< /katex >}}

where {{< katex >}}\sigma{{< /katex >}} is the sigmoid function. The loss raises the winner's score relative to the loser's; it does not regress onto the task reward values.

By default, both the backbone and the head are trained; `reward_freeze_backbone=True` trains only the head.

## Stage 2: Policy Reinforcement Learning

After Stage 1, {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}} is fixed. The policies generate {{< katex >}}G{{< /katex >}} new joint responses per task (`num_generations`), {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}} scores them without gradients, and the selected backend updates the policies with these shared rewards. With the default `rl_algorithm="magrpo"`, the advantage of response {{< katex >}}g{{< /katex >}} is its reward minus the group mean:

{{< katex display=true >}}
\widetilde{A}^{(g)}=\sum_{t=0}^{H-1}\left[\widetilde{R}_{\psi}(\mathbf{h}_t^{(g)},\mathbf{a}_t^{(g)})-\frac{1}{G}\sum_{g'=1}^{G}\widetilde{R}_{\psi}(\mathbf{h}_t^{(g')},\mathbf{a}_t^{(g')})\right].
{{< /katex >}}

Each agent {{< katex >}}i{{< /katex >}} weights its own log-probabilities {{< katex >}}\log\pi_{\theta_i}(a_{i,t}^{(g)}\mid h_{i,t}^{(g)}){{< /katex >}} by {{< katex >}}\widetilde{A}^{(g)}{{< /katex >}}, which the default `advantage_normalization=True` also divides by the group's standard deviation. Eq. 5 of the [MAPL paper](https://arxiv.org/abs/2609.32827) scales {{< katex >}}\widetilde{A}^{(g)}{{< /katex >}} by {{< katex >}}G/(G-1){{< /katex >}}, which equals the `"marloo"` advantage.

Choose the backend with `rl_algorithm`:

| Algorithm | Policy-training backend |
| --- | --- |
| `"magrpo"` | MAGRPO with mean-centered advantages; the default |
| `"mareinforce"` | REINFORCE with raw returns |
| `"maremax"` | MAGRPO-family updates with the maximum-reward baseline |
| `"marloo"` | MAGRPO-family updates with a leave-one-out baseline |
| `"maac"` | Multi-Agent Actor-Critic with a joint critic |
| `"iac"` | Independent Actor-Critic with per-actor critics or attached value heads |

The [Multi-Agent REINFORCE]({{< relref "/docs/marl/multi-agent-reinforce" >}}) and [Actor-Critic]({{< relref "/docs/marl/multi-agent-actor-critic" >}}) guides describe these updates; every backend uses {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}} in place of the task reward during training. This stage is controlled by `num_train_epochs` and `num_generations`, separately from `reward_num_train_epochs` and `preference_num_candidates`.

Optional frozen-reference KL regularization (`reference_kl_enabled`, `reference_kl_coef`, `reference_devices`) applies only to this stage. For the MAGRPO family, each agent's sampled KL estimate, scaled by `reference_kl_coef`, is subtracted from its returns before the baseline and normalization.

### Evaluation

For every backend, evaluation scores responses with the task `reward_func` (and `reward_processor`), not with {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}}, so it reports task scores rather than predicted rewards.

## Configuration

Import `MARLHFConfig` and `MARLHFTrainer` from `comlrl.trainers.preference`.

{{% hint info %}}
**MARLHFConfig** parameters:

- `num_agents`: Number of agents in decentralized mode, or task roles in centralized mode
- `collaboration_mode`: `"decentralized"` (default) or `"centralized"`
- `preference_num_candidates`: Candidates sampled per task and actor for preference collection; at least 2
- `preference_pairs_per_sample`: Maximum pairs kept per task, or `None` for no limit
- `pair_selection`: `"reward_gap"` keeps the largest reward gaps, `"random"` a random subset, and `"all"` every non-tied pair; `"all"` requires `preference_pairs_per_sample=None`
- `max_new_tokens`, `temperature`, `top_p`, `top_k`: Candidate generation settings
- `reward_model_name`: Model identifier for the reward backbone; defaults to the actor model identifier when available
- `reward_model_device`: Reward-model device; defaults to `"cuda:0"`, or `None` to use the trainer device
- `reward_learning_rate`, `reward_num_train_epochs`, `reward_train_batch_size`: Reward-model optimization settings
- `reward_max_length`: Optional token limit for the combined reward-model input
- `reward_freeze_backbone`: Train only the scalar reward head when `True`
- `rl_algorithm`: Online policy-training algorithm; defaults to `"magrpo"`
- `num_generations`: Online samples per prompt and actor (the group size); at least 2 for the MAGRPO family, or at least 1 for MAAC/IAC
- `num_train_epochs`, `agent_learning_rate`: Online policy-training epochs and learning rate
- `rollout_buffer_size`, `train_batch_size`: Online rollout buffer and update batch sizes; `train_batch_size` defaults to `rollout_buffer_size`
- `parallel_training`, `agent_devices`: Policy execution mode and device placement
- `critic_model_name`, `critic_learning_rate`, `critic_devices`: Critic settings for an actor-critic backend
- `critic_type`: `"v"` or `"q"` for an actor-critic backend
- `use_separate_critic`: Use separate critics or actor-attached value heads for IAC
- `value_loss_coef`, `value_clip_range`: Critic loss weight and IAC shared-value-head clipping; see [Actor-Critic]({{< relref "/docs/marl/multi-agent-actor-critic" >}})
- `reference_kl_enabled`, `reference_kl_coef`, `reference_devices`: Optional frozen-reference regularization for the online RL stage
- `eval_interval`, `eval_num_samples`, `eval_batch_size`: Evaluation schedule and sample sizes for the selected backend
- `logging_steps`: Training logging interval

`MARLHFConfig` inherits the shared preference settings from `MADPOConfig`; MARLHF does not use `dpo_beta`.
{{% /hint %}}

{{% hint info %}}
**MARLHFTrainer** setup:

- `agent_model` or `agents`: Model identifier for homogeneous agents, or a list of model identifiers or instances
- `num_agents`: Number of agents or task roles; must match `args.num_agents`
- `tokenizer`: One tokenizer, or one per actor
- `train_dataset`: Task items for preference collection and online policy training
- `eval_dataset`: Optional evaluation dataset
- `reward_func`: Required callable returning one shared task reward per aligned joint response
- `reward_processor`: Optional post-processing of task rewards during preference collection and evaluation
- `formatters`: One prompt formatter shared by all roles, or a list of role-specific formatters
- `args`: `MARLHFConfig` instance
- `model_config`: Optional model loading configuration
- `wandb_config`: Optional Weights & Biases configuration
- `eval_logger`, `eval_aggregator`: Optional evaluation metrics hooks, as in `MAGRPOTrainer`
- `centralized_comparator_adapter`: Prompt builder and response parser for centralized actors or comparators
- `critic_model` or `critics`: Optional critic sources for the MAAC/IAC backend. `critic_model` must be a model identifier string; pass loaded model instances through `critics`, as a one-element list for MAAC or one entry per actor for separate-critic IAC. Without explicit sources, `critic_model_name` or the actor model identifier is used
- `metrics_callback`: Optional custom metrics callback for the actor-critic backend
{{% /hint %}}

{{% hint warning %}}
Set `reward_model_name` when the actors are model objects without a model identifier. MAAC and separate-critic IAC likewise need explicit critic sources or a usable critic model identifier.
{{% /hint %}}

## Collaboration and Iterative Training

Decentralized MARLHF trains separate actors against one joint reward model. With `collaboration_mode="centralized"`, one actor generates the complete joint response; keep `num_agents` equal to the number of task roles and supply one actor and tokenizer. {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}} scores the joint text, which is parsed into role responses only for task rewards and evaluation. See [Training Framework]({{< relref "/docs/mapl/training-framework#collaboration-modes" >}}) for the layouts and adapter interface.

`MARLHFIterTrainer` repeats preference collection, reward learning, and policy RL. Each iteration {{< katex >}}\nu{{< /katex >}} with training pairs fits a newly initialized reward model on the selected pairs {{< katex >}}\mathcal{B}^{(\nu)}{{< /katex >}} and then runs Stage 2. New comparisons are labeled with the task reward by default; `preference_scoring_reward="reward_model"` uses the most recently fitted reward model instead, falling back to the task reward until one exists. An iteration is a refresh-and-training round, not a dialog turn.

See [Training Framework]({{< relref "/docs/mapl/training-framework#iterative-training" >}}) for the iterative configuration, the comparator, and the replay buffer that selects {{< katex >}}\mathcal{B}^{(\nu)}{{< /katex >}}.
