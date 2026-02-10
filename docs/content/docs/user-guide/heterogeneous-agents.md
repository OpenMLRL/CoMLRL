---
title: Heterogeneous Agents
linkTitle: Heterogeneous Agents
weight: 4
---

CoMLRL supports training teams where each agent uses a different base model. Specify per-agent model identifiers with a top-level `agents` list. The list length must match `num_agents`.

If you also set `model.name`, it is treated as the homogeneous default. When **both** are provided, they must be consistent (all entries in `agents` equal to `model.name`), otherwise CoMLRL raises an error.

{{% hint warning %}}
Tokenizers are loaded per agent by default. If your models use incompatible vocabularies, training may fail (e.g., in shared-critic settings). Prefer models from the same family or ensure tokenizer compatibility.
{{% /hint %}}

## Example: YAML (Top-Level Agents)

```yaml
agents:
  - "Qwen/Qwen2.5-Coder-3B"
  - "Qwen/Qwen3-4B-Instruct"

model:
  name: null

magrpo:
  num_agents: 2
```

## Example: CLI Overrides

```bash
--override \
agents='["Qwen/Qwen2.5-Coder-3B","Qwen/Qwen3-4B-Instruct"]' \
magrpo.num_agents=2
```
