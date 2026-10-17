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

Multi-agent PbRL improves collaborating agents by comparing joint responses to the same task. CoMLRL provides two methods: **MARLHF** fits a joint reward model to preference pairs and then trains the policies with RL, while **MADPO** optimizes the policies directly from the pairs. `MARLHFTrainer`, `MADPOTrainer`, and their iterative variants `MARLHFIterTrainer` and `MADPOIterTrainer` are available from `comlrl.trainers.preference`.

{{% hint warning %}}
All four trainers currently require `num_turns=1` and aligned joint responses (`joint_mode="aligned"`). An iteration of the iterative trainers is a round of data collection and training, not a dialog turn.
{{% /hint %}}

## Preference Pairs

Each agent {{< katex >}}i\in\mathcal{I}=\{1,\ldots,n\}{{< /katex >}} samples {{< katex >}}a_{i,t}\sim\pi_{\theta_i}(\cdot\mid h_{i,t}){{< /katex >}} from its local history, and the responses form the joint action {{< katex >}}\mathbf{a}_t=\{a_{1,t},\ldots,a_{n,t}\}{{< /katex >}} with joint history {{< katex >}}\mathbf{h}_t{{< /katex >}}. With a single turn ({{< katex >}}H=1{{< /katex >}}), {{< katex >}}h_{i,0}{{< /katex >}} is agent {{< katex >}}i{{< /katex >}}'s prompt and {{< katex >}}\mathbf{a}_0{{< /katex >}} is the joint response.

Users provide task items, not preference data; the trainers generate and score the pairs themselves. For each task item, every agent samples `preference_num_candidates` responses. Responses with the same candidate index form a joint response and share one reward from `reward_func` (after the optional `reward_processor`). Of two joint responses with different rewards, the higher one is the winner {{< katex >}}w{{< /katex >}} and the other the loser {{< katex >}}l{{< /katex >}}; ties are discarded. Each pair {{< katex >}}(\mathbf{h}_H^{w},\mathbf{h}_H^{l}){{< /katex >}} contains the two joint histories of one task, and `pair_selection` with `preference_pairs_per_sample` decides which pairs each task keeps.

The base trainers collect the pairs once, before training. If no task yields a non-tied pair, training returns without updating the policies.

## MARLHF

Multi-Agent Reinforcement Learning from Human Feedback (MARLHF) fits a joint reward model to the preference pairs (Stage 1), then trains the policies with RL on the learned reward (Stage 2). Because the reward model scores the team's complete response, separate actors learn from one shared reward.

{{< rollout-diagram diagram="marlhf-training" title="MARLHF training flow" description="Preference pairs fit a joint reward model, whose frozen scores then train both policies with RL." >}}

### Stage 1: Joint Reward Learning

The `JointRewardModel` {{< katex >}}\widetilde{R}_{\psi}(\mathbf{h}_t,\mathbf{a}_t){{< /katex >}} reads all agents' prompts and responses as one text and applies a scalar head to the causal-LM backbone's last hidden state at the final non-padding token. It minimizes the Bradley–Terry loss over the training pairs {{< katex >}}\mathcal{B}{{< /katex >}}:

{{< katex display=true >}}
\mathcal{L}_{\mathrm{RM}}(\psi)
=-\mathbb{E}_{(\mathbf{h}_H^{w},\mathbf{h}_H^{l})\sim\mathcal{B}}
\left[\log\sigma\left(\sum_{t=0}^{H-1}\Big[\widetilde{R}_{\psi}(\mathbf{h}_t^{w},\mathbf{a}_t^{w})-\widetilde{R}_{\psi}(\mathbf{h}_t^{l},\mathbf{a}_t^{l})\Big]\right)\right],
{{< /katex >}}

where {{< katex >}}\sigma{{< /katex >}} is the sigmoid function. The loss raises the winner's score relative to the loser's; it does not regress onto the task rewards. `reward_freeze_backbone=True` trains only the head. [Training Entities]({{< relref "/docs/user-guide/training-entities#llm-reward-models" >}}) describes how the reward model is loaded.

### Stage 2: Policy Reinforcement Learning

After Stage 1, {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}} is fixed. The policies generate {{< katex >}}G{{< /katex >}} new joint responses per task (`num_generations`), {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}} scores them without gradients, and the selected backend updates the policies with these shared rewards. With the default `rl_algorithm="magrpo"`, the advantage of response {{< katex >}}g{{< /katex >}} is its reward minus the group mean:

{{< katex display=true >}}
\widetilde{A}^{(g)}=\sum_{t=0}^{H-1}\left[\widetilde{R}_{\psi}(\mathbf{h}_t^{(g)},\mathbf{a}_t^{(g)})-\frac{1}{G}\sum_{g'=1}^{G}\widetilde{R}_{\psi}(\mathbf{h}_t^{(g')},\mathbf{a}_t^{(g')})\right].
{{< /katex >}}

Each agent {{< katex >}}i{{< /katex >}} weights its own log-probabilities {{< katex >}}\log\pi_{\theta_i}(a_{i,t}^{(g)}\mid h_{i,t}^{(g)}){{< /katex >}} by {{< katex >}}\widetilde{A}^{(g)}{{< /katex >}}, which the default `advantage_normalization=True` also divides by the group's standard deviation. Equation 5 of [Liu et al. (2026)](https://arxiv.org/abs/2609.32827) scales {{< katex >}}\widetilde{A}^{(g)}{{< /katex >}} by {{< katex >}}G/(G-1){{< /katex >}}, which equals the `"marloo"` advantage.

| `rl_algorithm` | Policy-training backend |
| --- | --- |
| `"magrpo"` (default) | MAGRPO, with a group-mean baseline |
| `"mareinforce"` | MAREINFORCE, with raw returns |
| `"marloo"` | MARLOO, with a leave-one-out baseline |
| `"maremax"` | MAReMax, with a group-maximum baseline |
| `"maac"` | MAAC, with a centralized critic |
| `"iac"` | IAC, with per-actor critics or attached value heads |

[Multi-Agent REINFORCE]({{< relref "/docs/marl/multi-agent-reinforce" >}}) and [Multi-Agent Actor-Critic]({{< relref "/docs/marl/multi-agent-actor-critic" >}}) describe these updates; every backend uses {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}} in place of the task reward. `num_train_epochs` and `num_generations` control this stage, separately from `reward_num_train_epochs` and `preference_num_candidates`.

Optional frozen-reference KL regularization (`reference_kl_enabled`, `reference_kl_coef`, `reference_devices`) applies only to this stage. For the MAGRPO family, each agent's sampled KL estimate, scaled by `reference_kl_coef`, is subtracted from its returns before the baseline and normalization.

Evaluation scores responses with the task `reward_func` (and `reward_processor`), not with {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}}, so it reports task scores rather than predicted rewards.

## MADPO

Multi-Agent Direct Preference Optimization (MADPO) trains the policies directly from the preference pairs, without a reward model or an RL stage. Its loss combines all agents' preference margins, so each policy learns from the team's outcome.

{{< rollout-diagram diagram="madpo-training" title="MADPO training flow" description="The agents' margins on each preference pair add up to one joint loss that updates every policy directly." >}}

### Joint Preference Objective

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

Equation 6 of [Liu et al. (2026)](https://arxiv.org/abs/2609.32827) uses the log-ratio {{< katex >}}\log(\pi_{\theta_i}/\pi_{\mathrm{ref},i}){{< /katex >}} against a frozen reference policy instead. CoMLRL implements the reference-free form above; the inherited `reference_kl_enabled`, `reference_kl_coef`, and `reference_devices` settings add no reference terms to the MADPO loss.

{{% hint warning %}}
The decentralized scorer excludes the last completion token after trimming padding, so each agent's margin omits that token's log-probability. The centralized scorer covers the full joint completion.
{{% /hint %}}

Evaluation samples responses from the current policies and scores them with the task reward. The training metrics `turn_1/reward_mean` and `turn_1/expected_return` instead average the candidate rewards recorded when the pairs were collected, so they do not reflect the updated policies.

## Collaboration Modes

`collaboration_mode` sets the trainable actor layout, and `comparator_generation_mode` sets the comparator layout in iterative training.

| Actor layout | Comparator layout | Joint responses |
| --- | --- | --- |
| Decentralized | Decentralized | Each role generates its own response on both sides of the comparison. |
| Decentralized | Centralized | Separate actors are compared with one comparator that generates all roles' responses. |
| Centralized | Centralized | One actor and one comparator each generate a complete joint response. |

The diagram shows the three layouts, with the actor on the left and the comparator {{< katex >}}\varpi{{< /katex >}} on the right.

{{< rollout-diagram diagram="mapl-layouts" title="Actor and comparator layouts" description="One policy per role on both sides; one policy per role against a comparator whose joint response is split into roles; or one policy on each side writing the whole joint response." >}}

In **decentralized** mode (the default), each agent has its own policy and response: MADPO sums the agents' margins, and the MARLHF reward model scores their joint response. In **centralized** mode, one actor {{< katex >}}\boldsymbol{\pi}_{\boldsymbol{\theta}}(\mathbf{a}_t\mid\mathbf{h}_t){{< /katex >}} receives every role's prompt and generates the complete joint response; {{< katex >}}\Delta_{\boldsymbol{\theta}}{{< /katex >}} is then that actor's margin on the joint text, and {{< katex >}}\widetilde{R}_{\psi}{{< /katex >}} scores the joint text.

### Centralized Preference Collaboration

With `collaboration_mode="centralized"`, one actor and one optimizer train on the complete joint response, including the role delimiters, so later roles are conditioned on earlier ones. The default `TaggedCentralizedComparatorAdapter` writes indexed `<agent_0>...</agent_0>` sections. A custom `centralized_comparator_adapter` can redefine the joint prompt (`build_prompt`) and the role parser (`parse_completion`) for both the actor and the comparator. The joint text is parsed into role responses only for task rewards and evaluation; replay, reward-model scoring, and training use the joint text.

- `num_agents` must equal the number of task roles, and the trainer takes one actor: `agent_model`, or one entry in `agents` with its tokenizer.
- `agent_devices` and, with reference KL, `reference_devices` describe that actor; the reward model and an external comparator keep their own device settings.
- `max_new_tokens` is the **total joint-response** budget, roughly twice a per-role limit for two roles. For MARLHF, `reward_max_length` must also cover the joint prompt and every role's output.
- MADPO remains reference-free; the MARLHF RL stage applies the configured KL penalty.
- The iterative trainers pair a centralized actor with a centralized comparator, which requires `comparator_centralized_agent_index=0`; a decentralized comparator is not supported.
- One joint completion counts as one environment step, and a preference pair uses `environment_steps_per_pair` (default 2). Actor checkpoints contain one actor (`agent_0`).

W&B records `collaboration_mode`, `num_roles`, `num_actor_models`, and the effective comparator generation mode.

## Iterative Training

`MADPOIterTrainer` and `MARLHFIterTrainer` alternate preference collection with training. Each round compares new samples from the current policies with [comparator](#comparator) samples at the same candidate index; either side can win, and ties are skipped. The round's pairs are stored as {{< katex >}}\mathcal{D}^{(\nu)}{{< /katex >}}, and the [replay buffer](#replay-buffer) selects the training pairs {{< katex >}}\mathcal{B}^{(\nu)}{{< /katex >}}.

{{< rollout-diagram diagram="mapl-iterative" title="One iterative training round" description="The task reward turns each comparison with the comparator into a preference pair, replay picks training pairs from this and earlier rounds, and MADPO, or a reward model followed by RL, produces the next policies." >}}

- **MADPOIter** runs MADPO updates on the selected pairs for `num_train_epochs` per iteration.
- **MARLHFIter** fits a newly initialized reward model on the selected pairs, then runs Stage 2. New comparisons are labeled with the task reward by default; `preference_scoring_reward="reward_model"` uses the most recently fitted reward model once one exists.

### Comparator

The comparator supplies the alternative responses in each comparison. `comparator_policy` chooses its source, and [`comparator_generation_mode`](#comparator-generation) chooses whether it generates separate role responses or one joint response. Every same-index comparison with a clear task-reward winner becomes a preference pair.

{{< rollout-diagram diagram="comparator-pairs" title="Comparator pairs" description="The current actors and the comparator sample for the same task; the task reward decides each same-index comparison in either direction, and the decided comparisons form the iteration's preference pairs." >}}

| Policy | Response source |
| --- | --- |
| `current` (default) | Active actor weights, sampled with a separate random-number stream. |
| `current_copy` | Current actor weights, copied to `comparator_devices` for each collection phase when those differ from the actor devices. |
| `model` | Explicit models from `comparator_model_name` or `comparator_agents`. |
| `history` | A saved actor checkpoint from `comparator_history_k` iterations back; the initial policy until one exists. |
| `api` | A remote model endpoint. |

The `history` source saves policy checkpoints automatically. The two sides may sample different numbers of candidates; only indices present on both sides are compared. `pair_selection="comparator_reward"` ranks comparisons by comparator reward and then reward gap; it does not assume the comparator wins.

{{% hint info %}}
**Comparator** parameters:

- `comparator_policy`: `"current"`, `"current_copy"`, `"model"`, `"history"`, or `"api"`
- `comparator_num_candidates`: Comparator sample count; defaults to `preference_num_candidates`
- `comparator_model_name` or `comparator_agents`: Model sources for `"model"`
- `comparator_history_k`: Iterations back for `"history"`; defaults to 1
- `comparator_devices`: Device placement for local comparator models
- `policy_checkpoint_dir`: Optional actor-checkpoint location; setting it also saves checkpoints for the other sources
{{% /hint %}}

#### Comparator Generation

With decentralized actors, a decentralized comparator (the default) generates one response per role. A centralized comparator receives the adapter's joint prompt and generates all roles' responses together; its joint text is parsed into role responses before pairs are built. With a centralized actor, the comparator's joint text is kept throughout.

{{% hint info %}}
**Comparator generation** parameters:

- `comparator_generation_mode`: `"decentralized"` or `"centralized"`
- `comparator_centralized_agent_index`: Role whose actor or checkpoint generates a centralized comparator's joint response (current, current-copy, or history sources); defaults to 0, and centralized actor training requires 0
- `centralized_comparator_adapter`: Trainer argument with the shared joint prompt builder and role parser
{{% /hint %}}

#### Local Models

`comparator_model_name` specifies one shared model source and `comparator_agents` a list of sources; either can be a model repository or a local checkpoint, as described in [Training Entities]({{< relref "/docs/user-guide/training-entities" >}}). A decentralized comparator uses one model per role. A centralized comparator uses one model, chosen by `comparator_centralized_agent_index` when `comparator_agents` lists one source per role.

```text
comparator_policy=model
comparator_model_name=/path/to/comparator-checkpoint
comparator_generation_mode=centralized
comparator_centralized_agent_index=0
comparator_devices=cuda:1
```

#### API Comparators

With `comparator_policy=api`, the comparator queries a remote model. `comparator_api_format` selects the request format:

- `generic` (default): The request includes the prompt, actor index, task item, candidate count, and generation settings; completions are read from `comparator_api_response_field`.
- `openai`: Chat Completions, also used by compatible endpoints such as DeepSeek's.
- `anthropic`: Anthropic Messages API (`https://api.anthropic.com/v1/messages`, `ANTHROPIC_API_KEY`).
- `openai_responses` (alias `responses` or `codex`): OpenAI Responses API (`https://api.openai.com/v1/responses`, `OPENAI_API_KEY`).

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

The replay buffer selects the stored pairs used for training in each round. The current round's dataset is saved first, so it is eligible alongside earlier rounds.

| Mode | Eligible preference data |
| --- | --- |
| `current` (default) | All pairs from the current round, without historical sampling. |
| `nearest_k` | The latest `preference_replay_k` non-empty round datasets, including the current one. |
| `all_history` | Every non-empty round dataset of the current trainer run. |
| `lambda_decay` | The same history, weighted toward recent rounds. |

Each row of the diagram shows which round datasets a mode can draw from and how strongly each is weighted, with the newest round on the right.

{{< rollout-diagram diagram="replay-modes" title="Replay modes" description="Which of five round datasets each replay mode draws from, and with what weight, before the training pairs are sampled by those weights." >}}

`nearest_k` means the most recent rounds, not nearest-neighbor retrieval. In `nearest_k` and `all_history`, eligible datasets have equal weight regardless of size, and historical modes sample from them rather than training on every stored pair.

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

`current` rejects `preference_replay_sample_size` and `preference_replay_lambda`, and accepts `preference_replay_k` only as `None` or 1.

```python
from comlrl.trainers.preference import MADPOIterConfig

args = MADPOIterConfig(
    num_agents=2,
    preference_replay_mode="lambda_decay",
    preference_replay_lambda=0.8,
    preference_replay_sample_size=256,
)
```

`MARLHFIterConfig` accepts the same options. To replay only the latest rounds, users can set `preference_replay_mode="nearest_k"` and the number of rounds through `preference_replay_k`.

#### Saved Preference Data

Each round is written to `iteration_0001.json`, `iteration_0002.json`, and so on under `preference_replay_dir`, with the prompts, preferred and rejected completions, rewards, and metadata. Historical samples are re-tokenized from this text. Setting `preference_replay_dir` to an earlier run alone does not import its files. [Training checkpoints]({{< relref "/docs/user-guide/training-entities#training-state" >}}) restore the replay index and its data. Actor checkpoints for the `history` comparator are stored separately under `policy_checkpoint_dir`.

## Configuration

<span id="api-marlhfconfig"></span>

{{% hint info %}}
**MARLHFConfig** parameters:

- `num_agents`: Number of agents in decentralized mode, or task roles in centralized mode
- `collaboration_mode`: `"decentralized"` (default) or `"centralized"`
- `preference_num_candidates`: Candidates sampled per task and actor for preference collection; at least 2
- `preference_pairs_per_sample`: Maximum pairs retained per task, or `None` for no limit
- `pair_selection`: `"reward_gap"` selects pairs with the largest reward gaps, `"random"` selects a random subset, and `"all"` retains every non-tied pair; `"all"` requires `preference_pairs_per_sample=None`
- `max_new_tokens`, `temperature`, `top_p`, `top_k`: Candidate generation settings
- `reward_model_name`: Model identifier for the reward backbone; defaults to the actor model identifier when available
- `reward_model_device`: Reward-model device; defaults to `"cuda:0"`, or `None` to use the trainer device
- `reward_learning_rate`, `reward_num_train_epochs`, `reward_train_batch_size`: Reward-model optimization settings
- `reward_max_length`: Optional token limit for the combined reward-model input
- `reward_freeze_backbone`: Trains only the scalar reward head when `True`
- `rl_algorithm`: Online policy-training algorithm; defaults to `"magrpo"`
- `num_generations`: Online samples per prompt and actor (the group size); at least 2 for the MAGRPO family, or at least 1 for MAAC/IAC
- `num_train_epochs`, `agent_learning_rate`: Online policy-training epochs and learning rate
- `rollout_buffer_size`, `train_batch_size`: Online rollout buffer and update batch sizes; `train_batch_size` defaults to `rollout_buffer_size`
- `parallel_training`, `agent_devices`: Policy execution mode and device placement
- `critic_model_name`, `critic_learning_rate`, `critic_devices`: Critic settings for an actor-critic backend
- `critic_type`: `"v"` or `"q"` for an actor-critic backend
- `use_separate_critic`: Choice of separate critics or actor-attached value heads for IAC
- `value_loss_coef`, `value_clip_range`: Critic loss weight and IAC shared-value-head clipping, as described in [Multi-Agent Actor-Critic]({{< relref "/docs/marl/multi-agent-actor-critic" >}})
- `reference_kl_enabled`, `reference_kl_coef`, `reference_devices`: Optional frozen-reference regularization for the online RL stage
- `eval_interval`, `eval_num_samples`, `eval_batch_size`: Evaluation schedule and sample sizes for the selected backend
- `logging_steps`: Training logging interval

`MARLHFConfig` inherits the shared preference settings from `MADPOConfig`; MARLHF does not use `dpo_beta`.
{{% /hint %}}

<span id="api-marlhftrainer"></span>

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
- `critic_model` or `critics`: Optional critic sources for the MAAC/IAC backend. `critic_model` must be a model identifier string. Users can supply loaded model instances through `critics`, using a one-element list for MAAC or one entry per actor for separate-critic IAC. Without explicit sources, `critic_model_name` or the actor model identifier is used
- `metrics_callback`: Optional custom metrics callback for the actor-critic backend
{{% /hint %}}

{{% hint warning %}}
`reward_model_name` is required when the actors are model objects without a model identifier. MAAC and separate-critic IAC likewise require explicit critic sources or a usable critic model identifier.
{{% /hint %}}

<span id="api-madpoconfig"></span>

{{% hint info %}}
**MADPOConfig** parameters:

- `num_agents`: Number of agents in decentralized mode, or task roles in centralized mode
- `collaboration_mode`: `"decentralized"` (default) or `"centralized"`
- `num_train_epochs`: Optimization epochs over the collected preference pairs
- `agent_learning_rate`: Policy learning rate
- `preference_num_candidates`: Candidates sampled per task and actor; at least 2
- `preference_pairs_per_sample`: Maximum pairs retained per task, or `None` for no limit
- `pair_selection`: `"reward_gap"` selects pairs with the largest reward gaps, `"random"` selects a random subset, and `"all"` retains every non-tied pair; `"all"` requires `preference_pairs_per_sample=None`
- `dpo_beta`: Positive coefficient on the joint margin
- `train_batch_size`: Preference pairs per update; defaults to `rollout_buffer_size`
- `max_new_tokens`, `temperature`, `top_p`, `top_k`: Candidate generation settings
- `parallel_training`, `agent_devices`: Policy execution mode and device placement
- `eval_interval`, `eval_num_samples`, `eval_batch_size`: Evaluation schedule and sample sizes; MADPO counts `eval_interval` in preference updates
- `logging_steps`: Training logging interval
{{% /hint %}}

<span id="api-madpotrainer"></span>

{{% hint info %}}
**MADPOTrainer** setup:

- `agent_model` or `agents`: Model identifier for homogeneous agents, or a list of model identifiers or instances
- `num_agents`: Number of agents or task roles; must match `args.num_agents`
- `tokenizer`: One tokenizer, or one per actor
- `train_dataset`: Task items from which preference candidates are generated
- `eval_dataset`: Optional evaluation dataset
- `reward_func`: Required callable returning one shared reward per aligned joint response
- `reward_processor`: Optional post-processing of task rewards
- `formatters`: One prompt formatter shared by all roles, or a list of role-specific formatters
- `args`: `MADPOConfig` instance
- `model_config`: Optional model loading configuration
- `wandb_config`: Optional Weights & Biases configuration
- `eval_logger`, `eval_aggregator`: Optional evaluation metrics hooks, as in `MAGRPOTrainer`
- `centralized_comparator_adapter`: Prompt builder and response parser for centralized actors or comparators
{{% /hint %}}

<span id="api-madpoiterconfig"></span>

{{% hint info %}}
**MADPOIterConfig** parameters:

- Inherits `MADPOConfig`; `num_train_epochs` applies to each iteration
- `num_iterations`: Number of collection-and-training rounds
- `pair_selection`: Adds `"comparator_reward"` (the iterative default), which prefers higher comparator rewards, then larger reward gaps; `"all"` retains every non-tied comparison and requires `preference_pairs_per_sample=None`
- Comparator and replay settings: described in [Comparator](#comparator) and [Replay Buffer](#replay-buffer)
{{% /hint %}}

<span id="api-marlhfiterconfig"></span>

{{% hint info %}}
**MARLHFIterConfig** parameters:

- Combines the MARLHF settings with the iterative comparator and replay settings
- `preference_scoring_reward`: `"task"` (default) labels new comparisons with the task reward; `"reward_model"` uses the previous iteration's reward model once one exists
- `reward_num_train_epochs`: Reward-model training epochs per iteration
{{% /hint %}}
