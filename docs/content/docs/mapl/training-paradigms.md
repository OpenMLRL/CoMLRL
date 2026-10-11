---
title: Training Paradigms
weight: 1
math: true
---

An actor generates the responses being learned; an iterative comparator supplies alternative responses to the same task. Choose their generation layouts separately from the [comparator's policy source]({{< relref "/docs/mapl/comparator" >}}). Both layouts can be used with the [MADPO and MARLHF objectives]({{< relref "/docs/mapl" >}}).

## Collaboration Modes

`collaboration_mode` determines the trainable actor layout. `comparator_generation_mode` determines the comparator layout in iterative training.

| Actor layout | Comparator layout | Joint responses |
| --- | --- | --- |
| Decentralized | Decentralized | Each role generates its own response on both sides of the comparison. |
| Decentralized | Centralized | Separate actors are compared with one comparator that writes all roles' responses. |
| Centralized | Centralized | One actor and one comparator each write a complete joint response. |

In **decentralized** mode, each agent has its own policy and generates its own response. MADPO combines their preference margins, while MARLHF learns one reward from the joint responses.

In **centralized** mode, one actor receives every role's prompt and generates one complete joint response. Keep `num_agents` equal to the number of task roles, but pass exactly one actor and tokenizer. Training and learned reward scoring retain the joint text; role parsing happens at task reward and evaluation boundaries. `max_new_tokens` is the total joint-response budget.

{{% hint warning %}}
All four preference trainers currently require `num_turns=1` and `joint_mode="aligned"`. Centralized actor training forces centralized comparator generation in the iterative configurations; a centralized actor with a decentralized comparator is not a separate supported layout.
{{% /hint %}}

## Centralized Preference Collaboration


`MADPOConfig`, `MARLHFConfig`, and their iterative variants accept
`collaboration_mode="centralized"` (default: `"decentralized"`). In this mode,
one trainable actor receives all role prompts and generates one joint response.
The complete response, including role delimiters, is optimized by one actor
and one optimizer. Later roles are causally conditioned on earlier roles.

The default `TaggedCentralizedComparatorAdapter` uses indexed role tags. To customize the joint prompt and response format, pass a downstream `centralized_comparator_adapter` implementing `build_prompt`
and `parse_completion`. The same adapter is used for the actor and comparator;
domain instructions and parsing stay outside CoMLRL. Parsing happens only at
the task reward and evaluation boundaries. Preference replay, learned reward
scoring, and policy training keep the complete joint text.

- Keep `num_agents` equal to the number of task roles (for example, 2). Use
  `agent_model` once, or supply exactly one entry in `agents` and its tokenizer.
- `agent_devices` describes one actor. The optional reward model and external
  comparator retain their independent device settings.
- Centralized actor training currently requires `num_turns=1`. If reference KL
  is enabled, `reference_devices` also describes one actor. MADPO retains its
  existing reference-free preference objective; MARLHF's online RL phase uses
  the configured KL penalty.
- Iterative trainers force `comparator_generation_mode="centralized"` and
  require `comparator_centralized_agent_index=0`. Current, current-copy, history,
  model, and API comparator sources retain their existing meanings.
- `max_new_tokens` is the **total joint response** budget, not a per-role
  budget. Consider doubling a previous per-role limit. For MARLHF, also allow
  enough `reward_max_length` for the joint prompt and both outputs.
- One joint completion still counts as one environment step; a preference pair
  uses `environment_steps_per_pair` (default 2). Role count does not multiply
  these counters. Checkpoints contain one actor (`agent_0`).

W&B records `collaboration_mode`, `num_roles`, `num_actor_models`, and the
effective comparator generation mode. Decentralized collaboration continues to
use the existing separate-actor generation, loss, and evaluation paths.


## Comparator Generation

For decentralized actors, select either `comparator_generation_mode="decentralized"` (the default) or `"centralized"`. A decentralized comparator generates one response per role. A centralized comparator receives the combined prompt and generates all role responses together.

The default `TaggedCentralizedComparatorAdapter` uses indexed `<agent_0>...</agent_0>` sections. A custom `centralized_comparator_adapter` defines the joint prompt through `build_prompt` and returns exactly one string per role through `parse_completion`.

When the actors are decentralized, the centralized comparator's joint text is parsed into role responses before constructing preference pairs. When the actor is also centralized, the joint text is retained throughout preference learning and is parsed only for task reward and evaluation.

{{% hint info %}}
**Comparator generation** parameters:

- `comparator_generation_mode`: Separate role responses (`"decentralized"`) or one joint response (`"centralized"`)
- `comparator_centralized_agent_index`: Actor or checkpoint role index used for centralized comparator generation with current, current-copy, or history policies; defaults to 0. It must be within the configured role range, and centralized actor training requires 0
- `centralized_comparator_adapter`: Trainer argument supplying the shared joint prompt builder and role parser

The comparator's model source, candidate count, and devices are documented in [Comparator]({{< relref "/docs/mapl/comparator" >}}).
{{% /hint %}}

## Iterative Training


`MADPOIterTrainer` and `MARLHFIterTrainer` alternate preference collection with training. Each collection round compares a new sample from the current policy with a sample from a comparator policy at the same candidate index. Either policy can supply the winner; reward ties are skipped.

- **MADPOIter** refreshes the pairs and runs MADPO updates for `num_train_epochs` per iteration.
- **MARLHFIter** refreshes the pairs, initializes and trains a reward model, and runs the selected online RL algorithm for each iteration.

Import the iterative configurations and trainers from `comlrl.trainers.preference`.


{{% hint info %}}
**MADPOIterConfig** parameters:

- Inherits `MADPOConfig`; `num_train_epochs` applies to each iteration
- `num_iterations`: Number of preference-refresh and training rounds
- `pair_selection`: Adds `"comparator_reward"` (the iterative default), which prioritizes higher comparator rewards, then larger reward gaps
- Comparator source and generation settings: see [Comparator]({{< relref "/docs/mapl/comparator" >}}) and [Comparator Generation](#comparator-generation)
- Replay selection, sample size, and storage settings: see [Replay Buffer]({{< relref "/docs/mapl/replay-buffer" >}})
{{% /hint %}}

The iterative `pair_selection="all"` retains all non-tied **index-aligned current-versus-comparator comparisons**, with `preference_pairs_per_sample=None`. An iteration refreshes preference data and then trains on the selected replay pairs; it is not a dialog turn.

{{% hint info %}}
**MARLHFIterConfig** parameters:

- Combines the MARLHF settings with the iterative comparator and replay settings above
- `preference_scoring_reward`: `"task"` (default) labels new comparisons with the task reward; `"reward_model"` uses the previous iteration's learned reward model once one is available, starting with task rewards in the first iteration
- `reward_num_train_epochs`: Reward-model training epochs per iteration
{{% /hint %}}
