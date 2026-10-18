---
title: Training Entities
linkTitle: Training Entities
weight: 2
tocDepth: 4
aliases:
  - /docs/user-guide/model-loading/
---

CoMLRL supports training LLM agents and facilitates their training with LLM-based critics or reward models. Agents and critics can be homogeneous or heterogeneous models from Hugging Face or local directories.
Users can specify [Hugging Face model identifiers](https://huggingface.co/models) through `agent_model`/`critic_model` for homogeneous setups, or provide `agents`/`critics` lists for heterogeneous setups. Trained entities can be saved and loaded into a new trainer for further training.


## LLM Agents

### Homogeneous Agents

Homogeneous agents are initialized by setting `agent_model` to a single model identifier and leaving `agents=None`. For trainers with a separate `num_agents` constructor argument, that argument can be set alongside the matching configuration value; the configuration alone does not override it.

The following example loads three _Qwen/Qwen2.5-1.5B_ agents:

```python
trainer = MAGRPOTrainer(
    agent_model="Qwen/Qwen2.5-1.5B",
    agents=None,
    num_agents=3,
    reward_func=reward_func,
    train_dataset=train_dataset,
    args=MAGRPOConfig(num_agents=3, num_turns=1, temperature=0.7, top_p=0.9, top_k=None),
)
```

### Heterogeneous Agents

Although homogeneous agents can be prompted to take on different roles and work on separate parts of a task in parallel, heterogeneous agents can contribute the specialized capabilities of different models.

Users can initialize heterogeneous agents by supplying model identifiers or loaded model objects in `agents`, with compatible tokenizers supplied through `tokenizer`.
With `agents` supplied, `agent_model` can remain `None`. If both are provided, their model names must match, and their list lengths must match the number of agents; otherwise, training raises an error.

The following example loads _Qwen/Qwen2.5-Coder-3B_ and _Qwen/Qwen2.5-Coder-7B_ agents:

```python
trainer = MAGRPOTrainer(
    agent_model=None,
    agents=["Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen2.5-Coder-7B"],
    reward_func=reward_func,
    train_dataset=train_dataset,
    args=MAGRPOConfig(num_agents=2, num_turns=1, temperature=0.7, top_p=0.9, top_k=None),
)
```

## LLM Critics

Algorithms such as Independent Actor-Critic (IAC) and Multi-Agent Actor-Critic (MAAC) use LLM critics. How these critics are loaded depends on the algorithm and the `use_separate_critic` setting. MARLHF trainers with `rl_algorithm="maac"` or `"iac"` accept the same `critic_model` and `critics` arguments, or `critic_model_name` in their configuration; otherwise, they use the first actor's model identifier.

### Centralized Critic

A separate centralized critic can be loaded through `critic_model` or a one-element `critics` list. It must be a supported causal language model: the trainer adds a value head and encodes the joint input with the first actor's tokenizer, so that tokenizer must also be compatible with the critic.

The following example pairs _Qwen/Qwen2.5-Coder-3B_ and _Qwen/Qwen2.5-Coder-1.5B_ actors with a centralized _Qwen/Qwen2.5-Coder-7B_ critic:

```python
trainer = MAACTrainer(
    agent_model=None,
    agents=["Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen2.5-Coder-1.5B"],
    tokenizer=[
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B"),
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B"),
    ],
    critic_model="Qwen/Qwen2.5-Coder-7B",
    critics=None,
    reward_func=reward_func,
    train_dataset=train_dataset,
    args=MAACConfig(num_agents=2, num_turns=1, temperature=0.7, top_p=0.9, top_k=None),
)
```

### Independent Critics

With independent critics, each agent can have its own critic. With `use_separate_critic=True`, users can supply one `critic_model` identifier to load a separate copy per agent, or a `critics` list of length `num_agents`. Each separate critic receives token IDs from its actor's tokenizer and must be compatible with it.
With `use_separate_critic=False`, each actor and critic share an LLM backbone and load together. In this case, `critic_model` and `critics` remain unset (`null` or `None`).
As with agents, supplying both `critic_model` and `critics` requires matching model names and lengths; otherwise, training raises an error.

The following example loads _Qwen/Qwen2.5-Coder-3B_ and _Qwen/Qwen2.5-Coder-7B_ actors with separate critics based on the same models:

```python
trainer = IACTrainer(
    agent_model=None,
    agents=["Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen2.5-Coder-7B"],
    tokenizer=[
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B"),
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B"),
    ],
    critic_model=None,
    critics=["Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen2.5-Coder-7B"],
    reward_func=reward_func,
    train_dataset=train_dataset,
    args=IACConfig(
        num_agents=2,
        num_turns=1,
        use_separate_critic=True,
        temperature=0.7,
        top_p=0.9,
        top_k=None,
    ),
)
```

Alternatively, each actor can share the same model with its critic:

```python
trainer = IACTrainer(
    agent_model=None,
    agents=["Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen2.5-Coder-7B"],
    tokenizer=[
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B"),
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B"),
    ],
    critic_model=None,
    critics=None,
    reward_func=reward_func,
    train_dataset=train_dataset,
    args=IACConfig(
        num_agents=2,
        num_turns=1,
        use_separate_critic=False,
        temperature=0.7,
        top_p=0.9,
        top_k=None,
    ),
)
```

{{% hint warning %}}
Trainers use `agents`/`critics` lists internally. The `agent_model` and `critic_model` parameters are shortcuts for homogeneous setups and must be consistent with any corresponding lists supplied.
{{% /hint %}}

{{% hint warning %}}
MAGRPO loads tokenizers automatically for model identifier strings. MAAC/IAC load a tokenizer automatically for `agent_model`, but require an explicit `tokenizer` when `agents` is supplied, even if it contains strings. Loaded model objects require one compatible shared tokenizer or a tokenizer list matching the actors. Heterogeneous actors may have different vocabularies when each uses its own compatible tokenizer; separate critics must also accept their actor's token IDs (the first actor's IDs for MAAC).
{{% /hint %}}

## LLM Reward Models

In PbRL, [MARLHF]({{< relref "/docs/marl/multi-llm-pbrl#marlhf" >}}) learns a joint reward model from preference pairs and keeps it fixed as the reward for policy learning. Its backbone is loaded through `reward_model_name` in `MARLHFConfig`, from a Hugging Face model identifier or a local directory of a causal language model. It defaults to the first actor's model identifier and is required when the actors are model objects without one. The trainer loads the backbone with its own tokenizer, adds a scalar head on the last non-padding token, and places the model on `reward_model_device` (default `"cuda:0"`; `None` uses the trainer device). `reward_freeze_backbone=True` trains only the head.

The following example loads a _Qwen/Qwen2.5-Coder-7B_ reward model for two _Qwen/Qwen2.5-Coder-3B_ agents:

```python
from comlrl.trainers.preference import MARLHFConfig, MARLHFTrainer

trainer = MARLHFTrainer(
    agent_model="Qwen/Qwen2.5-Coder-3B",
    num_agents=2,
    reward_func=reward_func,
    train_dataset=train_dataset,
    args=MARLHFConfig(
        num_agents=2,
        num_turns=1,
        reward_model_name="Qwen/Qwen2.5-Coder-7B",
        reward_model_device="cuda:1",
    ),
)
```

`MARLHFIterTrainer` reloads the reward model from this source at each iteration before reward learning. The task reward from `reward_func` still labels the preference pairs; [Training Initialization]({{< relref "/docs/user-guide/training-initialization#rewards" >}}) describes how to pass rule-based or LLM-based task rewards.

## Saving & Resuming Training

### Saving Models

Users can call `save_model(output_dir)` after training to explicitly save the actors' Hugging Face model configurations, weights, and tokenizers to local directories:

```python
trainer.train()
trainer.save_model("./saved-models/run-1")
```

### Training from Saved Models

Training can continue from a saved two-actor MAGRPO setup by supplying the actors' directories in the original role order. The parent `run-1` directory contains the actors; each `agent_i` directory is a loadable model.

```python
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer

continued_trainer = MAGRPOTrainer(
    agents=[
        "./saved-models/run-1/agent_0",
        "./saved-models/run-1/agent_1",
    ],
    num_agents=2,
    reward_func=reward_func,
    train_dataset=train_dataset,
    args=MAGRPOConfig(
        num_agents=2,
        num_turns=1,
        num_train_epochs=3,
        agent_learning_rate=5e-6,
        temperature=0.7,
        top_p=0.9,
        top_k=None,
    ),
)
continued_trainer.train()
continued_trainer.save_model("./saved-models/run-2")
```

MAGRPO loads tokenizers from the actor directories automatically. Here, `num_train_epochs=3` means three epochs in the new training run. The dataset, reward function, training configuration, and any custom formatters, reward processor, or external transition must be supplied again; `save_model` does not store them.

Other trainers support the same local-directory approach, subject to the tokenizer and actor-count requirements above. Preference trainers start preference collection and reward learning according to the new run's configuration.

IAC and MAAC save value heads separately as `value_head.pt`. Loading a critic or shared actor directory through the constructor restores the language-model backbone and creates a new value head. Users can retain the learned values by explicitly restoring the saved head before calling `train()`. Training checkpoints, unlike `save_model` exports, restore value heads automatically.

For a saved two-actor MAAC setup using the default linear value head:

```python
from pathlib import Path
import torch
from transformers import AutoTokenizer
from comlrl.trainers.actor_critic import MAACConfig, MAACTrainer

saved_dir = Path("./saved-models/maac-run-1")
actor_dirs = [str(saved_dir / f"agent_{i}") for i in range(2)]

continued_maac = MAACTrainer(
    agents=actor_dirs,
    tokenizer=[AutoTokenizer.from_pretrained(path) for path in actor_dirs],
    critic_model=str(saved_dir / "critic"),
    reward_func=reward_func,
    train_dataset=train_dataset,
    args=MAACConfig(num_agents=2, num_turns=1, num_train_epochs=3),
)
continued_maac.critics[0].value_head.load_state_dict(
    torch.load(
        saved_dir / "critic" / "value_head.pt",
        map_location="cpu",
        weights_only=True,
    )
)
continued_maac.train()
continued_maac.save_model("./saved-models/maac-run-2")
```

For IAC, the same `load_state_dict` step applies to `continued_iac.agents[i].value_head` when `use_separate_critic=False`, or to `continued_iac.critics[i].value_head` when it is `True`. Separate critic backbones also require loading through `critic_model` or `critics`.

The value-head architecture must match the saved run. IAC requires the original `IACConfig.value_head_hidden_dim` for shared heads and `IACConfig.critic_value_head_hidden_dim` for separate heads. A custom MAAC head requires the original `model_config={"critic_value_head_hidden_dim": ...}` in the trainer constructor. Continuing the same value-learning setup also requires the original `critic_type` (`"v"` or `"q"`).

### Training State

Users can enable training checkpoints through `train(checkpoint_dir=...)` and resume a compatible trainer with `resume_from_checkpoint`:

```python
# Save a checkpoint at each completed training boundary.
trainer.train(checkpoint_dir="./checkpoints", checkpoint_interval=1)

# Recreate the trainer with the same models, tokenizers, data, and callbacks.
resumed_trainer.train(
    resume_from_checkpoint="./checkpoints/checkpoint-1",
    checkpoint_dir="./checkpoints",
    checkpoint_interval=1,
)
```

Checkpoints preserve model and optimizer states, training progress, random states, and, where applicable, PbRL data and history.

Checkpoints are saved at completed epochs, iterations, or PbRL stages, with a final save at completion. `checkpoint_interval` controls periodic saves. Resuming repeats any work since the last checkpoint.

`num_train_epochs` and `num_iterations` count the total run, including completed work, and can be increased when resuming. Models, tokenizers, data, and training settings must remain compatible. Stateful datasets and callbacks can provide `state_dict()` and `load_state_dict()`; external service state is not restored. If a PbRL output directory already contains files, resumed output goes into a `resume-*` subdirectory.

For W&B, the saved run identity can be merged into the existing `wandb_config` before the trainer is constructed. This preserves logging and output settings and avoids opening an empty run during construction:

```python
from comlrl.utils import read_checkpoint_metadata

checkpoint = "./checkpoints/checkpoint-1"
identity = read_checkpoint_metadata(checkpoint)["wandb"]
if identity and "id" not in wandb_config and "resume" not in wandb_config:
    wandb_config = {**wandb_config, **identity, "resume": "allow"}

# Pass wandb_config to the trainer constructor before resuming.
```

Explicit `id` or `resume` settings take precedence. If the checkpoint contains no W&B identity, the helper returns `None` for `wandb` and the existing configuration is kept.

{{% hint info %}}
Checkpointing is disabled by default: `train()` without checkpoint options saves no training checkpoints. `save_model()` exports models only; it does not save the full state needed to resume training.
{{% /hint %}}
