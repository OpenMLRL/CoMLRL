---
title: Heterogeneous Agents
linkTitle: Heterogeneous Agents
weight: 4
---

CoMLRL supports training teams where each agent uses a different base model. Specify per-agent model identifiers with a top-level `agents` list. The list length must match `num_agents`.

If you also set `agent_model.name`, it is treated as the homogeneous default. When **both** are provided, they must be consistent (all entries in `agents` equal to `agent_model.name`), otherwise CoMLRL raises an error.

Internally, trainers always work with `agents`/`critics` lists. `agent_model` and `critic_model` are convenience shorthands that get expanded during initialization.

{{% hint warning %}}
Tokenizers are loaded per agent by default. If your models use incompatible vocabularies, training may fail (e.g., in shared-critic settings). Prefer models from the same family or ensure tokenizer compatibility.
{{% /hint %}}

## Example Overrides

Heterogeneous agents (per-agent models, disable `agent_model.name`):

```bash
--override \
agents='["Qwen/Qwen2.5-Coder-3B","Qwen/Qwen3-4B-Instruct"]' \
agent_model.name=None \
magrpo.num_agents=2
```

Homogeneous agents (single model for all agents):

```bash
--override \
agent_model.name="Qwen/Qwen2.5-Coder-3B" \
magrpo.num_agents=2
```

Homogeneous agents (explicit list, must match `agent_model.name`):

```bash
--override \
agents='["Qwen/Qwen2.5-Coder-3B","Qwen/Qwen2.5-Coder-3B"]' \
agent_model.name="Qwen/Qwen2.5-Coder-3B" \
magrpo.num_agents=2
```

IAC with a separate critic (heterogeneous agents):

```bash
--override \
agents='["Qwen/Qwen2.5-Coder-3B","Qwen/Qwen3-4B-Instruct"]' \
agent_model.name=None \
iac.use_separate_critic=true \
critic_model.name="Qwen/Qwen2.5-Coder-3B" \
iac.num_agents=2
```

MAAC (critic required, heterogeneous agents):

```bash
--override \
agents='["Qwen/Qwen2.5-Coder-3B","Qwen/Qwen3-4B-Instruct"]' \
agent_model.name=None \
critic_model.name="Qwen/Qwen2.5-Coder-3B" \
maac.num_agents=2
```

IAC with per-agent critics (list):

```bash
--override \
agents='["Qwen/Qwen2.5-Coder-3B","Qwen/Qwen3-4B-Instruct"]' \
agent_model.name=None \
iac.use_separate_critic=true \
critics='["Qwen/Qwen2.5-Coder-3B","Qwen/Qwen3-4B-Instruct"]' \
iac.num_agents=2
```

MAAC with an explicit shared critic list (single entry):

```bash
--override \
agents='["Qwen/Qwen2.5-Coder-3B","Qwen/Qwen3-4B-Instruct"]' \
agent_model.name=None \
critics='["Qwen/Qwen2.5-Coder-3B"]' \
maac.num_agents=2
```
