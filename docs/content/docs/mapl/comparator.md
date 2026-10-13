---
title: Comparator
weight: 2
---

A comparator supplies candidate responses for `MADPOIterTrainer` and `MARLHFIterTrainer`. The trainer compares current-policy and comparator responses at the same candidate index. Either side can win; tied rewards do not form a preference pair.

Choose the source with `comparator_policy`. Choose whether it generates separate role responses or one joint response in [Training Paradigms]({{< relref "/docs/mapl/training-paradigms#comparator-generation" >}}).

## Policy Sources

| Policy | Response source |
| --- | --- |
| `current` | Active actor weights, sampled with a separate comparator random-number stream. |
| `current_copy` | Current actor weights, with optional placement on separate comparator devices. |
| `model` | Explicit model identifiers or model sources supplied for the comparator. |
| `history` | A saved earlier actor checkpoint. |
| `api` | A remote model endpoint. |

`current` is the default. `current_copy` uses the same policy weights; when comparator and actor devices match, it reuses the actor models. With separate devices, the comparator copies are prepared for that iteration's collection phase and released afterward.

For `history`, `comparator_history_k=1` selects the preceding iteration's saved policy. Before a sufficiently old checkpoint exists, the trainer falls back to the initial policy. Policy checkpoints are saved automatically for this comparator mode.

{{% hint info %}}
**Comparator** parameters:

- `comparator_policy`: `"current"`, `"current_copy"`, `"model"`, `"history"`, or `"api"`
- `comparator_num_candidates`: Comparator sample count; defaults to `preference_num_candidates`
- `comparator_model_name` or `comparator_agents`: Required model source(s) for `comparator_policy="model"`
- `comparator_history_k`: How many iterations back to select a policy checkpoint; defaults to 1 for `"history"`
- `comparator_devices`: Device placement for local comparator models
- `policy_checkpoint_dir`: Optional actor-checkpoint output location; specifying it also enables checkpoint saving for other comparator modes
{{% /hint %}}

The current-policy and comparator sample counts can differ, but comparisons only use candidate indices available on both sides. `pair_selection="comparator_reward"` prioritizes comparisons with higher comparator rewards and then larger reward gaps. It does not assume that the comparator is always the preferred response.

## Local Models

Use `comparator_model_name` for a shared model source, or `comparator_agents` for explicit sources. A decentralized comparator uses one model per actor. A centralized comparator uses one model selected for the joint prompt; `comparator_agents` may contain one source or one source per actor, in which case `comparator_centralized_agent_index` selects the source.

Model identifiers can point to a compatible model repository or a local checkpoint. The common model-loading settings are described in [Model Loading]({{< relref "/docs/user-guide/model-loading" >}}).

Example local comparator overrides:

```text
comparator_policy=model
comparator_model_name=/path/to/comparator-checkpoint
comparator_generation_mode=centralized
comparator_centralized_agent_index=0
comparator_devices=cuda:1
```

## API Comparators


Iterative MADPO and MARLHF can use a remote model with `comparator_policy=api`.
The existing `comparator_api_format=openai` path remains compatible with
OpenAI-style Chat Completions endpoints such as DeepSeek. Two native formats
are also supported:

- `anthropic`: Anthropic Messages API. Use `https://api.anthropic.com/v1/messages`,
  `ANTHROPIC_API_KEY`, and a model such as `claude-sonnet-4-20250514`.
- `openai_responses` (or `responses`/`codex`): OpenAI Responses API. Use
  `https://api.openai.com/v1/responses`, `OPENAI_API_KEY`, and a model such as
  `gpt-5.1-codex`.

The native APIs are queried once per candidate because they do not use the
Chat Completions `n` parameter. The prompt remains the same centralized or
decentralized comparator prompt, and the trainer extracts text from the
provider-specific response format.

Example Anthropic comparator overrides:

```text
comparator_policy=api
comparator_api_format=anthropic
comparator_api_url=https://api.anthropic.com/v1/messages
comparator_api_model=claude-sonnet-4-20250514
comparator_api_key_env=ANTHROPIC_API_KEY
```

Example OpenAI/Codex comparator overrides:

```text
comparator_policy=api
comparator_api_format=openai_responses
comparator_api_url=https://api.openai.com/v1/responses
comparator_api_model=gpt-5.1-codex
comparator_api_key_env=OPENAI_API_KEY
```


## API Parameters

{{% hint info %}}
**Comparator API** parameters:

- `comparator_api_url`: Required endpoint URL when `comparator_policy="api"`
- `comparator_api_format`: `"generic"` (default), `"openai"` for Chat Completions, `"anthropic"` for Messages, or `"openai_responses"` for Responses
- `comparator_api_model`: Model name passed to the remote endpoint
- `comparator_api_key_env`: Environment variable from which to read the API key
- `comparator_api_key`: Optional explicitly supplied key
- `comparator_api_headers`: Additional request headers
- `comparator_api_key_header`, `comparator_api_key_prefix`: Authentication header and prefix; defaults are `"Authorization"` and `"Bearer"`, with native Anthropic header handling
- `comparator_api_timeout`: Request timeout in seconds; defaults to 120
- `comparator_api_max_n_per_request`: Maximum candidates requested in each Chat Completions call; useful for endpoints that limit `n`
- `comparator_api_response_field`: Field used for completions in a generic response; defaults to `"completions"`
- `comparator_api_extra_body`: Additional request-body fields
{{% /hint %}}

For the generic format, the request includes the prompt, actor index, task item, candidate count, and generation settings. The response parser reads `comparator_api_response_field`. Native formats use their provider-specific payload and text extraction.
