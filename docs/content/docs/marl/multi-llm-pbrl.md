---
title: Multi-LLM PbRL
weight: 3
math: true
tocDepth: 3
aliases:
  - /docs/mapl/
  - /docs/mapl/training-framework/
  - /docs/mapl/marlhf/
  - /docs/mapl/madpo/
  - /docs/mapl/training-paradigms/
  - /docs/mapl/comparator/
  - /docs/mapl/replay-buffer/
  - /docs/user-guide/multi-agent-preference-learning/
---

Multi-agent PbRL trains collaborating LLM agents from preferences between their joint responses. CoMLRL builds the preference data from task items and learns from it with one of two algorithms: **MARLHF** fits a joint reward model and then trains the policies with RL, while **MADPO** optimizes the policies directly from the preferences. `MARLHFTrainer`, `MADPOTrainer`, and their iterative variants `MARLHFIterTrainer` and `MADPOIterTrainer` are available from `comlrl.trainers.preference`.

{{% hint warning %}}
All four trainers currently require `num_turns=1` and aligned joint responses (`joint_mode="aligned"`). An iteration of the iterative trainers is a round of data collection and training, not a dialog turn.
{{% /hint %}}

## Preference Data

Each agent {{< katex >}}i\in\mathcal{I}=\{1,\ldots,n\}{{< /katex >}} samples {{< katex >}}a_{i,t}\sim\pi_{\theta_i}(\cdot\mid h_{i,t}){{< /katex >}} from its local history, and the responses form the joint action {{< katex >}}\mathbf{a}_t=\{a_{1,t},\ldots,a_{n,t}\}{{< /katex >}} with joint history {{< katex >}}\mathbf{h}_t{{< /katex >}}. With a single turn ({{< katex >}}H=1{{< /katex >}}), {{< katex >}}h_{i,0}{{< /katex >}} is agent {{< katex >}}i{{< /katex >}}'s prompt and {{< katex >}}\mathbf{a}_0{{< /katex >}} is the joint response.

Users provide task items, not preference data. The trainers sample joint responses, score them with an oracle function, and turn the scores into preference pairs.

### Oracle Function

The oracle function scores each joint response with one shared value. By default, it is the task reward `reward_func`, after the optional `reward_processor`; it can follow rules or call an LLM reward model, as described in [Training Initialization]({{< relref "/docs/user-guide/training-initialization#rewards" >}}). Both algorithms learn only from which joint response wins, so the score values matter only for choosing pairs. In `MARLHFIterTrainer`, `preference_scoring_reward="reward_model"` makes the most recently fitted reward model the oracle for new comparisons, with the task reward as a fallback until one exists.

### Preference Criterion

For each task item, every agent samples `preference_num_candidates` responses to its prompt, and responses with the same candidate index form a joint response. Of any two joint responses with different scores, the higher-scored one is preferred: it is the winner {{< katex >}}w{{< /katex >}}, and the other is the loser {{< katex >}}l{{< /katex >}}. Ties are discarded. `pair_selection` and `preference_pairs_per_sample` decide which pairs each task keeps. In iterative training, each comparison is instead between the current agents' joint response and the [comparator](#comparator)'s joint response at the same candidate index, and either side can win.

{{% hint info %}}
**Preference Data** parameters:

- `preference_num_candidates`: Candidates sampled per task and actor; at least 2 (default 80)
- `preference_pairs_per_sample`: Maximum pairs kept per task, or `None` for no limit (default 16)
- `pair_selection`: `"reward_gap"` (default) keeps the pairs with the largest reward gaps, `"random"` a random subset, and `"all"` every non-tied pair, which requires `preference_pairs_per_sample=None`; the iterative trainers add `"comparator_reward"`, their default, which prefers higher comparator rewards and then larger reward gaps
- `max_new_tokens`, `temperature`, `top_p`, `top_k`: Candidate generation settings
{{% /hint %}}

### Data Format

A pair {{< katex >}}(\mathbf{h}_H^{w},\mathbf{h}_H^{l}){{< /katex >}} holds the two joint histories of one task. Each pair is a record like the following, with one entry per agent in each list:

```json
{
  "prompts": ["<agent 1 prompt>", "<agent 2 prompt>"],
  "winner_completions": ["<agent 1 response>", "<agent 2 response>"],
  "loser_completions": ["<agent 1 response>", "<agent 2 response>"],
  "winner_reward": 0.92,
  "loser_reward": 0.41,
  "candidate_reward_mean": 0.63,
  "raw_rewards": [0.92, 0.41],
  "target_raw_reward": 0.92,
  "comparator_raw_reward": 0.41
}
```

`winner_reward` and `loser_reward` are the oracle scores of the two joint responses, and `candidate_reward_mean` is the mean score of the candidates compared for the task. The last three fields come from iterative training: `raw_rewards` holds the winner's and loser's scores before `reward_processor`, and `target_raw_reward` and `comparator_raw_reward` hold those of the current agents' sample and the comparator's sample. The iterative trainers write each round's pairs to `iteration_0001.json`, `iteration_0002.json`, and so on under `preference_replay_dir`, as `{"iteration": 1, "num_pairs": ..., "pairs": [...]}`.

## Training Pipeline

Training a cooperative LLM-based multi-agent system with PbRL alternates between collecting preference data from the current agents and learning from it with MARLHF or MADPO.

- **Base trainers** (`MARLHFTrainer`, `MADPOTrainer`) collect the pairs once, then train on them. If no task yields a non-tied pair, training returns without updating the policies.
- **Iterative trainers** (`MARLHFIterTrainer`, `MADPOIterTrainer`) repeat the cycle for `num_iterations` rounds. Round {{< katex >}}\nu{{< /katex >}} compares the current agents with the [comparator](#comparator), stores the new pairs as {{< katex >}}\mathcal{D}^{(\nu)}{{< /katex >}}, lets the [replay buffer](#replay-buffer) select the training pairs {{< katex >}}\mathcal{B}^{(\nu)}{{< /katex >}} from this and earlier rounds, and trains on them. MADPO runs its updates for `num_train_epochs` per round. MARLHF fits a newly initialized reward model on them and then runs policy learning.

{{< rollout-diagram diagram="mapl-iterative" title="One iterative training round" description="The task reward turns each comparison with the comparator into a preference pair, replay picks training pairs from this and earlier rounds, and MADPO, or a reward model followed by RL, produces the next policies." >}}

Evaluation always samples from the current policies and scores them with the task reward (`reward_func` and `reward_processor`), never with a learned reward model.

{{% hint info %}}
**PbRL Trainer** parameters:

- `agent_model` or `agents`: Model identifier for homogeneous agents, or a list of model identifiers or instances
- `num_agents`: Number of agents or task roles; must match `args.num_agents`
- `tokenizer`: One tokenizer, or one per actor
- `train_dataset`: Task items for preference collection and, in MARLHF, policy learning
- `eval_dataset`: Optional evaluation dataset
- `reward_func`: Required callable returning one shared task reward per aligned joint response
- `reward_processor`: Optional post-processing of task rewards
- `formatters`: One prompt formatter shared by all roles, or a list of role-specific formatters
- `args`: The trainer's configuration instance
- `model_config`: Optional model loading configuration
- `wandb_config`: Optional Weights & Biases configuration
- `eval_logger`, `eval_aggregator`: Optional evaluation metrics hooks, as in `MAGRPOTrainer`
- `centralized_comparator_adapter`: Joint prompt builder and role parser for a centralized actor or comparator; the default writes indexed `<agent_0>...</agent_0>` sections
- `critic_model` or `critics` (MARLHF only): Optional critic sources for a MAAC or IAC backend. `critic_model` must be a model identifier string. Users can supply loaded model instances through `critics`, using a one-element list for MAAC or one entry per actor for separate-critic IAC. Without explicit sources, `critic_model_name` or the actor model identifier is used
- `metrics_callback` (MARLHF only): Optional custom metrics callback for an actor-critic backend
{{% /hint %}}

{{% hint info %}}
**MADPOIterConfig** parameters:

- Inherits `MADPOConfig`; `num_train_epochs` applies to each iteration
- `num_iterations`: Number of collection-and-training rounds (default 6)
- Comparator and replay settings: described in [Comparator](#comparator) and [Replay Buffer](#replay-buffer)
{{% /hint %}}

{{% hint info %}}
**MARLHFIterConfig** parameters:

- Combines the MARLHF settings with `num_iterations` and the comparator and replay settings
- `preference_scoring_reward`: `"task"` (default) labels new comparisons with the task reward; `"reward_model"` uses the previous iteration's reward model once one exists
- `reward_num_train_epochs`: Reward-model training epochs per iteration (default 2)
{{% /hint %}}

### Comparator

In iterative training, the comparator supplies the other side of each comparison. The current agents and the comparator respond to the same task, and every same-index comparison with a clear winner under the oracle function becomes a pair, whichever side wins.

{{< rollout-diagram diagram="comparator-pairs" title="Comparator pairs" description="The current actors and the comparator sample for the same task; the task reward decides each same-index comparison in either direction, and the decided comparisons form the iteration's preference pairs." >}}

`comparator_policy` chooses the comparator's source:

{{% hint success %}}

- `comparator_policy="current"` (default): Active actor weights, sampled with a separate random-number stream.

- `comparator_policy="current_copy"`: Current actor weights, copied to `comparator_devices` for each collection phase when those differ from the actor devices.

- `comparator_policy="model"`: Explicit models from `comparator_model_name` or `comparator_agents`.

- `comparator_policy="history"`: A saved actor checkpoint from `comparator_history_k` iterations back; the initial policy until one exists.

- `comparator_policy="api"`: A remote model endpoint.
{{% /hint %}}

The `history` source saves policy checkpoints automatically. The two sides may sample different numbers of candidates; only indices present on both sides are compared, and `pair_selection="comparator_reward"` does not assume the comparator wins. By default, the comparator generates one response per role; with `comparator_generation_mode="centralized"`, one comparator model writes every role's response in a single joint response, which is parsed into role responses before pairs are built.

{{% hint info %}}
**Comparator** parameters:

- `comparator_policy`: `"current"`, `"current_copy"`, `"model"`, `"history"`, or `"api"`
- `comparator_num_candidates`: Comparator sample count; defaults to `preference_num_candidates`
- `comparator_model_name` or `comparator_agents`: One shared model source, or one per role, for `"model"`
- `comparator_history_k`: Iterations back for `"history"`; defaults to 1
- `comparator_devices`: Device placement for local comparator models
- `comparator_generation_mode`: `"decentralized"` (default) or `"centralized"`
- `comparator_centralized_agent_index`: Role whose actor, checkpoint, or model writes a centralized comparator's joint response; defaults to 0, which a centralized actor requires
- `policy_checkpoint_dir`: Optional actor-checkpoint location; setting it also saves checkpoints for the other sources
{{% /hint %}}

Model sources can be model repositories or local checkpoints, as described in [Training Entities]({{< relref "/docs/user-guide/training-entities" >}}):

```text
comparator_policy=model
comparator_model_name=/path/to/comparator-checkpoint
comparator_generation_mode=centralized
comparator_centralized_agent_index=0
comparator_devices=cuda:1
```

#### API Comparators

With `comparator_policy="api"`, the comparator queries a remote model. `comparator_api_format` selects the request format:

{{% hint success %}}

- `comparator_api_format="generic"` (default): The request includes the prompt, actor index, task item, candidate count, and generation settings; completions are read from `comparator_api_response_field`.

- `comparator_api_format="openai"`: Chat Completions, also used by compatible endpoints such as DeepSeek's.

- `comparator_api_format="anthropic"`: Anthropic Messages API (`https://api.anthropic.com/v1/messages`, `ANTHROPIC_API_KEY`).

- `comparator_api_format="openai_responses"` (alias `responses` or `codex`): OpenAI Responses API (`https://api.openai.com/v1/responses`, `OPENAI_API_KEY`).
{{% /hint %}}

The Anthropic and Responses formats have no `n` parameter, so they send one request per candidate. Requests use the standard centralized or decentralized comparator prompts.

```text
comparator_policy=api
comparator_api_format=anthropic
comparator_api_url=https://api.anthropic.com/v1/messages
comparator_api_model=claude-sonnet-4-20250514
comparator_api_key_env=ANTHROPIC_API_KEY
```

{{% hint info %}}
**Comparator API** parameters:

- `comparator_api_url`: Endpoint URL; required for `"api"`
- `comparator_api_format`: `"generic"` (default), `"openai"`, `"anthropic"`, or `"openai_responses"`
- `comparator_api_model`: Model name passed to the endpoint
- `comparator_api_key_env` or `comparator_api_key`: Environment variable holding the key, or the key itself
- `comparator_api_headers`: Additional request headers
- `comparator_api_key_header`, `comparator_api_key_prefix`: Authentication header and prefix; default `"Authorization"` and `"Bearer"`, with native Anthropic header handling
- `comparator_api_timeout`: Request timeout in seconds; default 120
- `comparator_api_max_n_per_request`: Maximum candidates per Chat Completions call, for endpoints that limit `n`
- `comparator_api_response_field`: Completions field in a generic response; default `"completions"`
- `comparator_api_extra_body`: Additional request-body fields
{{% /hint %}}

### Replay Buffer

The replay buffer selects the stored pairs used for training in each round. The current round's dataset is saved first, so it is eligible alongside earlier rounds. `preference_replay_mode` sets which rounds are eligible:

{{% hint success %}}

- `preference_replay_mode="current"` (default): All pairs from the current round, without historical sampling.

- `preference_replay_mode="nearest_k"`: The latest `preference_replay_k` non-empty round datasets, including the current one.

- `preference_replay_mode="all_history"`: Every non-empty round dataset of the current trainer run.

- `preference_replay_mode="lambda_decay"`: The same history, weighted toward recent rounds.
{{% /hint %}}

Each row of the diagram shows which round datasets a mode can draw from and how strongly each is weighted, with the newest round on the right.

{{< rollout-diagram diagram="replay-modes" title="Replay modes" description="Which of five round datasets each replay mode draws from, and with what weight, before the training pairs are sampled by those weights." >}}

`nearest_k` means the most recent rounds, not nearest-neighbor retrieval. In `nearest_k` and `all_history`, eligible datasets have equal weight regardless of size, and historical modes sample from them rather than training on every stored pair. Replayed pairs are re-tokenized from the saved round files. Setting `preference_replay_dir` to an earlier run alone does not import its files, while [training checkpoints]({{< relref "/docs/user-guide/training-entities#training-state" >}}) restore the replay index and its data.

#### Lambda Decay

With age {{< katex >}}a=0{{< /katex >}} for the newest of {{< katex >}}K{{< /katex >}} eligible datasets, `preference_replay_lambda` ({{< katex >}}\lambda{{< /katex >}}) sets the selection weights:

{{< katex display=true >}}
w_a = \frac{\lambda^a}{\sum_{j=0}^{K-1}\lambda^j},
\qquad 0 \leq \lambda \leq 1.
{{< /katex >}}

{{< katex >}}\lambda=0{{< /katex >}} keeps only the newest dataset, smaller values favor recent rounds, and {{< katex >}}\lambda=1{{< /katex >}} weights every dataset equally.

#### Sampling

Historical modes assign each of the `preference_replay_sample_size` draws to a dataset by weight, then sample pairs within each dataset. Sampling proceeds without repetition when the assigned count does not exceed the dataset size; larger counts cycle through shuffled copies. The combined sample is shuffled. The default sample size is the current round's pair count, or the latest non-empty dataset's size if the current round is empty.

{{% hint info %}}
**Replay Buffer** parameters:

- `preference_replay_mode`: `"current"` (default), `"nearest_k"`, `"all_history"`, or `"lambda_decay"`
- `preference_replay_k`: Number of recent non-empty datasets for `"nearest_k"`; required in that mode
- `preference_replay_lambda`: Decay parameter in [0, 1]; required for `"lambda_decay"`
- `preference_replay_sample_size`: Pairs sampled by the historical modes, or `None` for the default above
- `preference_replay_dir`: Optional location of the saved round datasets
{{% /hint %}}

`current` rejects `preference_replay_sample_size` and `preference_replay_lambda` and accepts `preference_replay_k` only as `None` or 1. The following configuration instead samples 256 pairs from all rounds, weighted toward recent ones:

```python
from comlrl.trainers.pbrl import MADPOIterConfig

args = MADPOIterConfig(
    num_agents=2,
    preference_replay_mode="lambda_decay",
    preference_replay_lambda=0.8,
    preference_replay_sample_size=256,
)
```

`MARLHFIterConfig` accepts the same options. To replay only the latest rounds, users can set `preference_replay_mode="nearest_k"` and the number of rounds through `preference_replay_k`.

## MARLHF

Multi-Agent Reinforcement Learning from Human Feedback (MARLHF) fits a joint reward model to the preference pairs, then trains the policies with RL on the learned reward. Because the reward model scores the team's complete response, separate actors learn from one shared reward.

{{< rollout-diagram diagram="marlhf-training" title="MARLHF training flow" description="Preference pairs fit a joint reward model, whose frozen scores then train both policies with RL." >}}

### Reward Learning

The `JointRewardModel` {{< katex >}}\widetilde{R}_{\psi}(\mathbf{h}_t,\mathbf{a}_t){{< /katex >}} reads all agents' prompts and responses as one text and applies a scalar head to the causal-LM backbone's last hidden state at the final non-padding token. It minimizes the Bradley–Terry loss over the training pairs {{< katex >}}\mathcal{B}{{< /katex >}}:

{{< katex display=true >}}
\mathcal{L}_{\mathrm{RM}}(\psi)
=-\mathbb{E}_{(\mathbf{h}_H^{w},\mathbf{h}_H^{l})\sim\mathcal{B}}
\left[\log\sigma\left(\sum_{t=0}^{H-1}\Big[\widetilde{R}_{\psi}(\mathbf{h}_t^{w},\mathbf{a}_t^{w})-\widetilde{R}_{\psi}(\mathbf{h}_t^{l},\mathbf{a}_t^{l})\Big]\right)\right],
{{< /katex >}}

where {{< katex >}}\sigma{{< /katex >}} is the sigmoid function. The loss raises the winner's score relative to the loser's; it does not regress onto the task rewards. [Training Entities]({{< relref "/docs/user-guide/training-entities#llm-reward-models" >}}) describes how the reward model is loaded.

### Policy Learning

After reward learning, {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}} is fixed and takes the place of the task reward. The policies sample new joint responses for each task, {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}} scores them without gradients, and a MARL backend updates each agent's policy to maximize the learned return:

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}_t \sim \boldsymbol{\pi}_{\boldsymbol{\theta}}}
\Bigg[\sum_{t=0}^{H-1} \widetilde{R}_{\psi}(\mathbf{h}_t,\mathbf{a}_t)\Bigg].
{{< /katex >}}

`rl_algorithm` selects the backend:

{{% hint success %}}

- `rl_algorithm="magrpo"` (default): MAGRPO, with a group-mean baseline.

- `rl_algorithm="mareinforce"`: MAREINFORCE, with raw returns.

- `rl_algorithm="marloo"`: MARLOO, with a leave-one-out baseline.

- `rl_algorithm="maremax"`: MAReMax, with a group-maximum baseline.

- `rl_algorithm="maac"`: MAAC, with a centralized critic.

- `rl_algorithm="iac"`: IAC, with per-actor critics or attached value heads.
{{% /hint %}}

[Multi-Agent REINFORCE]({{< relref "/docs/marl/multi-agent-reinforce" >}}) and [Multi-Agent Actor-Critic]({{< relref "/docs/marl/multi-agent-actor-critic" >}}) describe these updates, with {{< katex >}}\widetilde{R}_{\psi}(\mathbf{h}_t,\mathbf{a}_t){{< /katex >}} in place of the reward {{< katex >}}r_t{{< /katex >}}. Optional frozen-reference KL regularization (`reference_kl_enabled`, `reference_kl_coef`, `reference_devices`) applies only to policy learning: each agent's sampled KL estimate, scaled by `reference_kl_coef`, is subtracted from its reward signal.

{{% hint info %}}
**MARLHFConfig** parameters:

- Inherits the [MADPOConfig](#api-madpoconfig) and [preference data](#api-preference-data) settings, except `dpo_beta`, which MARLHF does not use
- `reward_model_name`: Model identifier for the reward backbone; defaults to the actor model identifier when available
- `reward_model_device`: Reward-model device; defaults to `"cuda:0"`, or `None` to use the trainer device
- `reward_learning_rate`, `reward_num_train_epochs`, `reward_train_batch_size`: Reward-model optimization settings
- `reward_max_length`: Optional token limit for the combined reward-model input
- `reward_freeze_backbone`: Trains only the scalar reward head when `True`
- `rl_algorithm`: Policy-learning backend; defaults to `"magrpo"`
- `num_generations`: Policy-learning samples per prompt and actor (the group size); at least 2 for the MAGRPO family, or at least 1 for MAAC/IAC
- `num_train_epochs`, `agent_learning_rate`: Policy-learning epochs and learning rate
- `rollout_buffer_size`, `train_batch_size`: Policy-learning rollout buffer and update batch sizes; `train_batch_size` defaults to `rollout_buffer_size`
- `critic_model_name`, `critic_learning_rate`, `critic_devices`: Critic settings for an actor-critic backend
- `critic_type`: `"v"` or `"q"` for an actor-critic backend
- `use_separate_critic`: Choice of separate critics or actor-attached value heads for IAC
- `value_loss_coef`, `value_clip_range`: Critic loss weight and IAC shared-value-head clipping, as described in [Multi-Agent Actor-Critic]({{< relref "/docs/marl/multi-agent-actor-critic" >}})
- `reference_kl_enabled`, `reference_kl_coef`, `reference_devices`: Optional frozen-reference regularization for policy learning
{{% /hint %}}

{{% hint warning %}}
`reward_model_name` is required when the actors are model objects without a model identifier. MAAC and separate-critic IAC likewise require explicit critic sources or a usable critic model identifier.
{{% /hint %}}

## MADPO

Multi-Agent Direct Preference Optimization (MADPO) trains the policies directly from the preference pairs, without a reward model or an RL stage. Its loss combines all agents' preference margins, so each policy learns from the team's outcome.

{{< rollout-diagram diagram="madpo-training" title="MADPO training flow" description="The agents' margins on each preference pair add up to one joint loss that updates every policy directly." >}}

Each agent's margin is the difference between its log-probabilities of the winning and losing responses, and the joint margin sums these across agents:

{{< katex display=true >}}
\Delta_{\theta_i}=\sum_{t=0}^{H-1}\Big(\log\pi_{\theta_i}(a_{i,t}^{w}\mid h_{i,t}^{w})-\log\pi_{\theta_i}(a_{i,t}^{l}\mid h_{i,t}^{l})\Big),
\qquad
\Delta_{\boldsymbol{\theta}}=\sum_{i\in\mathcal{I}}\Delta_{\theta_i}.
{{< /katex >}}

MADPO minimizes

{{< katex display=true >}}
\mathcal{L}_{\mathrm{MADPO}}(\boldsymbol{\theta})
=-\mathbb{E}_{(\mathbf{h}_H^{w},\mathbf{h}_H^{l})\sim\mathcal{B}}
\Big[\log\sigma\big(\eta\,\Delta_{\boldsymbol{\theta}}\big)\Big],
{{< /katex >}}

where {{< katex >}}\eta>0{{< /katex >}} is `dpo_beta` (default 0.1). The task reward only orders each pair; the loss uses the policies' log-probabilities. The gradient for agent {{< katex >}}i{{< /katex >}} is

{{< katex display=true >}}
\nabla_{\theta_i}\mathcal{L}_{\mathrm{MADPO}}(\boldsymbol{\theta})
=-\eta\,\mathbb{E}_{\mathcal{B}}
\Big[\sigma\big(-\eta\,\Delta_{\boldsymbol{\theta}}\big)\,\nabla_{\theta_i}\Delta_{\theta_i}\Big].
{{< /katex >}}

The weight {{< katex >}}\sigma(-\eta\,\Delta_{\boldsymbol{\theta}}){{< /katex >}} depends on the team's joint margin, while the gradient flows only through agent {{< katex >}}i{{< /katex >}}'s own margin. In each batch, the trainer computes all margins once without gradients, then updates each agent with its own differentiable margin plus the others' detached margins.

The objective is reference-free: the inherited `reference_kl_enabled`, `reference_kl_coef`, and `reference_devices` settings add no reference terms to the MADPO loss. The training metrics `turn_1/reward_mean` and `turn_1/expected_return` average the candidate rewards recorded when the pairs were collected, so they do not reflect the updated policies.

{{% hint warning %}}
The decentralized scorer excludes the last completion token after trimming padding, so each agent's margin omits that token's log-probability. The centralized scorer covers the full joint completion.
{{% /hint %}}

{{% hint info %}}
**MADPOConfig** parameters:

- `num_agents`: Number of agents, or of task roles with a centralized actor
- `collaboration_mode`: `"decentralized"` (default) trains one actor per agent; `"centralized"` trains one actor that writes the complete joint response, and `max_new_tokens` then covers the whole joint response
- `num_train_epochs`: Optimization epochs over the collected preference pairs
- `agent_learning_rate`: Policy learning rate
- `dpo_beta`: Positive coefficient on the joint margin (default 0.1)
- `train_batch_size`: Preference pairs per update; defaults to `rollout_buffer_size`
- `parallel_training`, `agent_devices`: Policy execution mode and device placement
- `eval_interval`, `eval_num_samples`, `eval_batch_size`: Evaluation schedule and sample sizes; MADPO counts `eval_interval` in preference updates
- `logging_steps`: Training logging interval
- `environment_steps_per_pair`: Environment steps counted per preference pair (default 2)
- The [preference data](#api-preference-data) settings
{{% /hint %}}
