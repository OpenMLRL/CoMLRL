---
title: Model Loading
linkTitle: Model Loading
weight: 2
---

CoMLRL supports both homogeneous and heterogeneous models.
Users can assign `agent_model`/`critic_model` with [HuggingFace model identifiers](https://huggingface.co/models) for homogeneous setups, or provide `agents`/`critics` lists for heterogeneous setups.

## Homogeneous Agents

The easiest way to start the journey of CoMLRL is to load `num_agents` homogeneous agents with a single model identifier.
Users can set `agent_model.name` to a single model identifier while keeping `agents: null`.

For example, to load 3 _Qwen/Qwen2.5-1.5B_ agents:

```python
trainer = MAGRPOTrainer(
    agent_model="Qwen/Qwen2.5-1.5B",
    agents=None,
    num_agents=3,
)
```

## Heterogeneous Agents

Although homogeneous LLM agents can be specified into different roles by prompting, using heterogeneous LLMs with different skills can further unleash the potential of multi-agent collaboration.
Users can load a list of heterogeneous agents in `agents`, where the length of the list should match `num_agents`. Each entry should specify a model identifier and optional tokenizer/model kwargs.
When `agents` is provided, `agent_model` should be set to null or ignored; if both are provided, they must match (same names, correct length) or training will raise an error.

For example, to load a _Qwen/Qwen2.5-Coder-3B_ and a _Qwen/Qwen2.5-Coder-7B_:

```python
trainer = MAGRPOTrainer(
    agent_model=None,
    agents=["Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen2.5-Coder-7B"],
    num_agents=2,
)
```

## Loading LLM Critics

The loading of critics depends on the algorithm and the `use_separate_critic` setting.
In Multi-Agent Actor-Critic (MAAC), a single separated centralized critic is used, so one model should be provided in `critic_model` or `critics` without any constraints on model types.

For example, to load a _Qwen/Qwen2.5-Coder-3B_ and a _Qwen/Qwen2.5-Coder-1.5B_ with a centralized _Qwen/Qwen2.5-Coder-7B_ critic:

```python
trainer = MAACTrainer(
    agent_model=None,
    agents=["Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen2.5-Coder-1.5B"],
    critic_model="Qwen/Qwen2.5-Coder-7B",
    critics=None,
    num_agents=2,
)
```

In Independent Actor-Critic (IAC), each agent can have its own critic. When `use_separate_critic=true`, users should provide `critic_model.name` or `critics` with length `num_agents` to load separate critics for each agent.
When `use_separate_critic=false`, each agent shares its LLM agent backbone with its critic, and the critic is loaded at the same time as the agent. In this case, `critic_model` and `critics` should not be provided and set to null or None.
Similarly to agents, if both `critic_model` and `critics` are provided, they must match (same names, correct length) or training will raise an error.

For example, to load a _Qwen/Qwen2.5-Coder-3B_ and a _Qwen/Qwen3-Coder-Next_ with separate critics of the same models:

```python
trainer = IACTrainer(
    agent_model=None,
    agents=["Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen3-Coder-Next"],
    critic_model=None,
    critics=["Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen3-Coder-Next"],
    num_agents=2,
)
```

Or actor can share the same model with its critic:

```python
trainer = IACTrainer(
    agent_model=None,
    agents=["Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen3-Coder-Next"],
    critic_model=None,
    critics=None,
    num_agents=2,
)
```

{{% hint warning %}}
Internally, trainers always work with `agents`/`critics` lists. `agent_model` and `critic_model` are convenience shortcuts for homogeneous settings; if both are provided, they must be consistent.
{{% /hint %}}

{{% hint warning %}}
Tokenizers are loaded per agent by default. If your models use incompatible vocabularies, training may fail (e.g., in shared-critic settings). Prefer models from the same family or ensure tokenizer compatibility.
{{% /hint %}}
