---
title: Training Framework
weight: 1
math: true
aliases:
  - /docs/mapl/training-paradigms/
  - /docs/mapl/comparator/
  - /docs/mapl/replay-buffer/
---

MARLHF and MADPO share one training framework. Actors generate joint responses, the task reward ranks them into preference pairs, and the iterative trainers repeat this against a comparator, with a replay buffer selecting the pairs for each update. This page covers the actor layouts, the iterative loop, the comparator, and replay; [MARLHF]({{< relref "/docs/mapl/marlhf" >}}) and [MADPO]({{< relref "/docs/mapl/madpo" >}}) describe the two learning objectives.

## Collaboration Modes

`collaboration_mode` sets the trainable actor layout, and `comparator_generation_mode` sets the comparator layout in iterative training.

| Actor layout | Comparator layout | Joint responses |
| --- | --- | --- |
| Decentralized | Decentralized | Each role generates its own response on both sides of the comparison. |
| Decentralized | Centralized | Separate actors are compared with one comparator that writes all roles' responses. |
| Centralized | Centralized | One actor and one comparator each write a complete joint response. |

The diagram compares the three supported layouts, with the actor on the left and the comparator {{< katex >}}\varpi{{< /katex >}} on the right.

{{< rollout-diagram diagram="mapl-layouts" title="Actor and comparator layouts" description="One policy per role on both sides; one policy per role against a comparator whose joint response is split into roles; or one policy on each side writing the whole joint response." >}}

In **decentralized** mode, each agent has its own policy and response; MADPO combines their preference margins, and MARLHF learns one reward from the joint responses. In **centralized** mode, one actor receives every role's prompt and writes the complete joint response.

{{% hint warning %}}
All four preference trainers currently require `num_turns=1` and `joint_mode="aligned"`. Centralized actor training forces a centralized comparator in the iterative trainers; a centralized actor with a decentralized comparator is not supported.
{{% /hint %}}

## Centralized Preference Collaboration

`MADPOConfig`, `MARLHFConfig`, and their iterative variants accept `collaboration_mode="centralized"` (default `"decentralized"`). One actor and one optimizer then train on the complete joint response, including the role delimiters, so later roles are conditioned on earlier ones.

The default `TaggedCentralizedComparatorAdapter` writes indexed `<agent_0>...</agent_0>` sections. A downstream `centralized_comparator_adapter` can redefine the joint prompt (`build_prompt`) and the role parser (`parse_completion`); the same adapter serves the actor and the comparator. Parsing happens only for the task reward and evaluation; replay, learned-reward scoring, and training keep the joint text.

- Keep `num_agents` equal to the number of task roles, but supply one actor: `agent_model` once, or one entry in `agents` with its tokenizer.
- `agent_devices` and, with reference KL, `reference_devices` describe one actor; the reward model and an external comparator keep their own device settings.
- `max_new_tokens` is the **total joint-response** budget, so roughly double a per-role limit. For MARLHF, `reward_max_length` must also cover the joint prompt and every role's output.
- MADPO stays reference-free; MARLHF's online RL stage applies the configured KL penalty.
- The iterative trainers require `comparator_centralized_agent_index=0`; every comparator source keeps its meaning.
- One joint completion counts as one environment step, and a preference pair uses `environment_steps_per_pair` (default 2). Checkpoints contain one actor (`agent_0`).

W&B records `collaboration_mode`, `num_roles`, `num_actor_models`, and the effective comparator generation mode.

## Iterative Training

`MADPOIterTrainer` and `MARLHFIterTrainer` alternate preference collection with training. Each round compares new samples from the current policy with [comparator](#comparator) samples at the same candidate index; either side can win, and ties are skipped. The round's pairs are stored as {{< katex >}}\mathcal{D}^{(\nu)}{{< /katex >}}, and the [replay buffer](#replay-buffer) selects the training pairs {{< katex >}}\mathcal{B}^{(\nu)}{{< /katex >}}.

The diagram unrolls one iteration {{< katex >}}\nu{{< /katex >}}, from sampling against the comparator to the updated policies.

{{< rollout-diagram diagram="mapl-iterative" title="One iterative training round" description="The task reward turns each comparison with the comparator into a preference pair, replay picks training pairs from this and earlier rounds, and MADPO, or a reward model followed by RL, produces the next policies." >}}

- **MADPOIter** runs MADPO updates on the selected pairs for `num_train_epochs` per iteration.
- **MARLHFIter** fits a newly initialized reward model on the selected pairs, then runs the selected online RL algorithm.

An iteration is a refresh-and-training round, not a dialog turn. Import the iterative configurations and trainers from `comlrl.trainers.preference`.

{{% hint info %}}
**MADPOIterConfig** parameters:

- Inherits `MADPOConfig`; `num_train_epochs` applies to each iteration
- `num_iterations`: Number of refresh-and-training rounds
- `pair_selection`: Adds `"comparator_reward"` (the iterative default), which prefers higher comparator rewards, then larger reward gaps; `"all"` keeps every non-tied comparison and requires `preference_pairs_per_sample=None`
- Comparator and replay settings: see [Comparator](#comparator) and [Replay Buffer](#replay-buffer)
{{% /hint %}}

{{% hint info %}}
**MARLHFIterConfig** parameters:

- Combines the MARLHF settings with the iterative comparator and replay settings
- `preference_scoring_reward`: `"task"` (default) labels new comparisons with the task reward; `"reward_model"` uses the previous iteration's reward model once one exists
- `reward_num_train_epochs`: Reward-model training epochs per iteration
{{% /hint %}}

## Comparator

The comparator supplies the other side of each comparison. `comparator_policy` chooses its source, and [`comparator_generation_mode`](#comparator-generation) chooses whether it writes separate role responses or one joint response.

Each iteration, the current actors and the comparator answer the same task, and every same-index comparison with a clear task-reward winner becomes a preference pair.

{{< rollout-diagram diagram="comparator-pairs" title="Comparator pairs" description="The current actors and the comparator sample for the same task; the task reward decides each same-index comparison in either direction, and the decided comparisons form the iteration's preference pairs." >}}

| Policy | Response source |
| --- | --- |
| `current` (default) | Active actor weights, sampled with a separate random-number stream. |
| `current_copy` | Current actor weights, copied to `comparator_devices` for each collection phase when those differ from the actor devices. |
| `model` | Explicit models from `comparator_model_name` or `comparator_agents`. |
| `history` | A saved actor checkpoint from `comparator_history_k` iterations back; the initial policy until one exists. |
| `api` | A remote model endpoint. |

The `history` source saves policy checkpoints automatically. The two sides may sample different numbers of candidates, but only indices present on both sides are compared. `pair_selection="comparator_reward"` ranks comparisons by comparator reward and then gap; it does not assume the comparator wins.

{{% hint info %}}
**Comparator** parameters:

- `comparator_policy`: `"current"`, `"current_copy"`, `"model"`, `"history"`, or `"api"`
- `comparator_num_candidates`: Comparator sample count; defaults to `preference_num_candidates`
- `comparator_model_name` or `comparator_agents`: Model sources for `"model"`
- `comparator_history_k`: Iterations back for `"history"`; defaults to 1
- `comparator_devices`: Device placement for local comparator models
- `policy_checkpoint_dir`: Optional actor-checkpoint location; setting it also saves checkpoints for the other sources
{{% /hint %}}

### Comparator Generation

With decentralized actors, a decentralized comparator (the default) writes one response per role, while a centralized comparator receives the adapter's joint prompt and writes all roles together. Its joint text is then parsed into role responses before pairs are built. With a centralized actor, the comparator's joint text is kept throughout preference learning.

{{% hint info %}}
**Comparator generation** parameters:

- `comparator_generation_mode`: `"decentralized"` or `"centralized"`
- `comparator_centralized_agent_index`: Role whose actor or checkpoint writes a centralized comparator's joint response (current, current-copy, or history sources); defaults to 0, and centralized actor training requires 0
- `centralized_comparator_adapter`: Trainer argument with the shared joint prompt builder and role parser
{{% /hint %}}

### Local Models

`comparator_model_name` gives one shared model source, and `comparator_agents` lists explicit sources. A decentralized comparator uses one model per role; a centralized comparator uses one model, chosen by `comparator_centralized_agent_index` when `comparator_agents` lists one source per role. Sources can be model repositories or local checkpoints; see [Model Loading]({{< relref "/docs/user-guide/model-loading" >}}).

```text
comparator_policy=model
comparator_model_name=/path/to/comparator-checkpoint
comparator_generation_mode=centralized
comparator_centralized_agent_index=0
comparator_devices=cuda:1
```

### API Comparators

With `comparator_policy=api`, the comparator queries a remote model. `comparator_api_format` selects the request format:

- `generic` (default): The request carries the prompt, actor index, task item, candidate count, and generation settings; completions are read from `comparator_api_response_field`.
- `openai`: Chat Completions, also used by compatible endpoints such as DeepSeek's.
- `anthropic`: Anthropic Messages API (`https://api.anthropic.com/v1/messages`, `ANTHROPIC_API_KEY`).
- `openai_responses` (alias `responses` or `codex`): OpenAI Responses API (`https://api.openai.com/v1/responses`, `OPENAI_API_KEY`).

The Anthropic and Responses formats have no `n` parameter, so they send one request per candidate. Prompts are the usual centralized or decentralized comparator prompts.

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

## Replay Buffer

The replay buffer decides which stored pairs train each round. The current round is written first, so it is eligible alongside earlier rounds.

| Mode | Eligible preference data |
| --- | --- |
| `current` (default) | All pairs from the current round, without historical sampling. |
| `nearest_k` | The latest `preference_replay_k` non-empty round datasets, including the current one. |
| `all_history` | Every non-empty round dataset of the current trainer run. |
| `lambda_decay` | The same history, weighted toward recent rounds. |

Each row of the diagram shows which round datasets a mode can draw from and how strongly each is weighted, with the newest round on the right.

{{< rollout-diagram diagram="replay-modes" title="Replay modes" description="Which of five round datasets each replay mode draws from, and with what weight, before the training pairs are sampled by those weights." >}}

`nearest_k` means the most recent rounds, not nearest-neighbor retrieval. In `nearest_k` and `all_history`, eligible datasets have equal weight regardless of size, and historical modes sample from them rather than training on every stored pair.

### Lambda Decay

With age {{< katex >}}a=0{{< /katex >}} for the newest of {{< katex >}}K{{< /katex >}} eligible datasets, `preference_replay_lambda` ({{< katex >}}\lambda{{< /katex >}}) sets the selection weights:

{{< katex display=true >}}
w_a = \frac{\lambda^a}{\sum_{j=0}^{K-1}\lambda^j},
\qquad 0 \leq \lambda \leq 1.
{{< /katex >}}

{{< katex >}}\lambda=0{{< /katex >}} keeps only the newest dataset, smaller values favor recent rounds, and {{< katex >}}\lambda=1{{< /katex >}} weights every dataset equally.

### Sampling

Historical modes assign each of the `preference_replay_sample_size` draws to a dataset by weight, then sample pairs within each dataset, without repetition while the assigned count fits (larger counts cycle through shuffled copies). The combined sample is shuffled. The default sample size is the current round's pair count, or the latest non-empty dataset's size if the current round is empty.

{{% hint info %}}
**Replay Buffer** parameters:

- `preference_replay_mode`: `"current"` (default), `"nearest_k"`, `"all_history"`, or `"lambda_decay"`
- `preference_replay_k`: Number of recent non-empty datasets for `"nearest_k"`; required in that mode
- `preference_replay_lambda`: Decay parameter in [0, 1]; required for `"lambda_decay"`
- `preference_replay_sample_size`: Pairs sampled by the historical modes, or `None` for the default above
- `preference_replay_dir`: Optional location of the saved round datasets
{{% /hint %}}

`current` rejects `preference_replay_sample_size` and `preference_replay_lambda`, and accepts `preference_replay_k` only as `None` or 1. Use `preference_replay_k` only with `nearest_k`, and `preference_replay_lambda` only with `lambda_decay`.

```python
from comlrl.trainers.preference import MADPOIterConfig

args = MADPOIterConfig(
    num_agents=2,
    preference_replay_mode="lambda_decay",
    preference_replay_lambda=0.8,
    preference_replay_sample_size=256,
)
```

`MARLHFIterConfig` accepts the same options; for the latest rounds only, use `preference_replay_mode="nearest_k"` with `preference_replay_k`.

### Saved Preference Data

Each round is written to `iteration_0001.json`, `iteration_0002.json`, and so on under `preference_replay_dir`, with the prompts, preferred and rejected completions, rewards, and metadata; historical samples are re-tokenized from this text. The replay index covers only the current trainer run, so pointing `preference_replay_dir` at an earlier run does not import its files. Actor checkpoints for the `history` comparator are stored separately under `policy_checkpoint_dir`.
