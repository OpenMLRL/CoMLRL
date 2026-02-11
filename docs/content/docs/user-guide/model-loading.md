---
title: Model Loading
linkTitle: Model Loading
weight: 2
---

CoMLRL supports both homogeneous and heterogeneous models.
Users can assign `agent_model`/`critic_model` with [HuggingFace model identifiers](https://huggingface.co/models) for homogeneous setups, or provide `agents`/`critics` lists for heterogeneous setups.

## Loading Homogeneous Agents

Users can set `agent_model.name` to a single model identifier while keeping `agents: null`. This loads `num_agents` instances of the same model.
Training with both `agent_model` and `agents` is allowed, but all entries in `agents` must match `agent_model.name` to avoid conflicts.

For example,

## Loading Heterogeneous Agents

Provide a list of model identifiers in `agents` with length equal to `num_agents`. Set `agent_model.name: null` to avoid conflicts. Generation settings in `agent_model` still apply as defaults for all agents.

## Loading Critics

Critic loading depends on the algorithm and `use_separate_critic`:

- **MAAC**: Provide `critic_model.name` (single model) or `critics` with one entry.
- **IAC with `use_separate_critic=true`**: Provide `critic_model.name` or `critics` with length `num_agents`.
- **IAC with `use_separate_critic=false`**: Do not provide critic models. Set `critic_model.name: null` and `critics: null`.
- **REINFORCE/MAGRPO**: No critics are used; keep `critic_model`/`critics` null.

If both `critic_model` and `critics` are provided, they must match (same names, correct length) or training will error.

{{% hint warning %}}
Internally, trainers always work with `agents`/`critics` lists. `agent_model` and `critic_model` are convenience shortcuts for homogeneous settings; if both are provided, they must be consistent.
{{% /hint %}}

{{% hint warning %}}
Tokenizers are loaded per agent by default. If your models use incompatible vocabularies, training may fail (e.g., in shared-critic settings). Prefer models from the same family or ensure tokenizer compatibility.
{{% /hint %}}
