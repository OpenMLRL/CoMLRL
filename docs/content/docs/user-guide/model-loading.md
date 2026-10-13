---
title: Model Loading & Saving
linkTitle: Model Loading & Saving
weight: 2
---

CoMLRL supports both homogeneous and heterogeneous models.
Users can assign `agent_model`/`critic_model` with [HuggingFace model identifiers](https://huggingface.co/models) for homogeneous setups, or provide `agents`/`critics` lists for heterogeneous setups.

Local model directories are also supported. You can save trained actors, load their weights into a new trainer, and train them further. The sections below cover model loading, the saved directory layout, and which training state must be initialized again.

The examples below assume you have defined a task-specific `reward_func` and `train_dataset`, as shown in the [quick start]({{< relref "docs/examples/comlrl-quick-start" >}}). They use `num_turns=1`; multi-turn training additionally requires an `external_transition`.

```python
from transformers import AutoTokenizer
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer
from comlrl.trainers.actor_critic import IACConfig, IACTrainer, MAACConfig, MAACTrainer
```

## Homogeneous Agents

The easiest way to start the journey of CoMLRL is to load `num_agents` homogeneous agents with a single model identifier.
Set `agent_model` to a model identifier string while keeping `agents=None`. For MAGRPO, pass the actor count directly as the trainer's `num_agents` argument; setting only `MAGRPOConfig.num_agents` does not change the number of homogeneous actors.

For example, to load 3 _Qwen/Qwen2.5-1.5B_ agents:

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

## Heterogeneous Agents

Although homogeneous LLM agents can be specified into different roles by prompting, using heterogeneous LLMs with different skills can further unleash the potential of multi-agent collaboration.
Pass model identifier strings or loaded model objects in `agents`, and pass compatible tokenizers separately through `tokenizer`. MAGRPO infers the actor count from `agents` when `agent_model=None`; MAAC/IAC require the list length to match `args.num_agents`. For MAGRPO, use a list of identifiers or a list of loaded objects, without mixing the two.
When `agents` is provided, set `agent_model=None`. If both are provided, they must match (same model names and correct length), otherwise training raises an error.

For example, to load a _Qwen/Qwen2.5-Coder-3B_ and a _Qwen/Qwen2.5-Coder-7B_:

```python
trainer = MAGRPOTrainer(
    agent_model=None,
    agents=["Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen2.5-Coder-7B"],
    reward_func=reward_func,
    train_dataset=train_dataset,
    args=MAGRPOConfig(num_agents=2, num_turns=1, temperature=0.7, top_p=0.9, top_k=None),
)
```

## Loading LLM Critics

The loading of critics depends on the algorithm and the `use_separate_critic` setting.
In Multi-Agent Actor-Critic (MAAC), one separate centralized critic is provided through `critic_model` or a one-element `critics` list. It must be a supported causal language model: the trainer adds a value head and encodes the joint input with the first actor's tokenizer, so that tokenizer must also be compatible with the critic.

For example, to load a _Qwen/Qwen2.5-Coder-3B_ and a _Qwen/Qwen2.5-Coder-1.5B_ with a centralized _Qwen/Qwen2.5-Coder-7B_ critic:

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

In Independent Actor-Critic (IAC), each agent can have its own critic. When `use_separate_critic=True`, provide one `critic_model` identifier to load a separate copy per agent, or a `critics` list of length `num_agents`. Each separate critic receives token IDs from its corresponding actor's tokenizer and must be compatible with it.
When `use_separate_critic=false`, each agent shares its LLM agent backbone with its critic, and the critic is loaded at the same time as the agent. In this case, `critic_model` and `critics` should not be provided and set to null or None.
Similarly to agents, if both `critic_model` and `critics` are provided, they must match (same names, correct length) or training will raise an error.

For example, to load a _Qwen/Qwen2.5-Coder-3B_ and a _Qwen/Qwen2.5-Coder-7B_ with separate critics of the same models:

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

Or actor can share the same model with its critic:

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
Internally, trainers always work with `agents`/`critics` lists. `agent_model` and `critic_model` are convenience shortcuts for homogeneous settings; if both are provided, they must be consistent.
{{% /hint %}}

{{% hint warning %}}
MAGRPO loads tokenizers automatically for model identifier strings. MAAC/IAC load a tokenizer automatically for `agent_model`, but require an explicit `tokenizer` when `agents` is supplied, even if it contains strings. For loaded model objects, provide one compatible shared tokenizer or a tokenizer list matching the actors. Heterogeneous actors may have different vocabularies when each uses its own compatible tokenizer; separate critics must also accept their actor's token IDs (the first actor's IDs for MAAC).
{{% /hint %}}

## Saving Models

Call `save_model(output_dir)` explicitly after training. It writes the actors' Hugging Face model configuration, weights, and tokenizers to local directories:

```python
trainer.train()
trainer.save_model("./saved-models/run-1")
```

The directory layout depends on the trainer. In the table, `output_dir` is the directory passed to `save_model`, and `i` is the zero-based actor index.

| Trainer | Actor and tokenizer | Additional saved weights |
| --- | --- | --- |
| MA-REINFORCE family, including MAGRPO | `output_dir/agent_i/` | None |
| MADPO, MARLHF, and their iterative variants | `output_dir/agent_i/` | None |
| IAC, one actor | `output_dir/` | Shared head: `value_head.pt`; separate critic: `critic/` with its own `value_head.pt` |
| IAC, multiple actors | `output_dir/agent_i/` | Shared head: `agent_i/value_head.pt`; separate critic: `agent_i/critic/` with its own `value_head.pt` |
| MAAC | `output_dir/agent_i/` | One centralized critic in `output_dir/critic/`, including `value_head.pt` |

Paths without an `output_dir` prefix are relative to `output_dir`. Critic directories contain the critic backbone configuration and weights; they use the corresponding actor's tokenizer, or the first actor's tokenizer for MAAC.

{{% hint info %}}
The MA-REINFORCE and preference trainers use `agent_0/` even with one actor. Centralized MAPL also saves one actor in `agent_0/`, while `num_agents` continues to describe the number of task roles.

MARLHF's inherited `save_model` exports the actors and tokenizers. It does not export the learned reward model, reward head, or critics created by its online actor-critic backend.
{{% /hint %}}

## Training from Saved Models

To continue training a saved two-actor MAGRPO setup, pass each actor's directory in the original role order. The parent `run-1` directory is a container for the actors; each `agent_i` directory is a loadable model.

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

MAGRPO loads the tokenizers from those actor directories automatically. Here, `num_train_epochs=3` means three epochs in the new training run. Supply the dataset, reward function, training configuration, and any custom formatters, reward processor, or external transition again; these are not stored by `save_model`.

Use the same local-directory approach with the other trainers, retaining their tokenizer and actor-count requirements described above. For centralized MAPL, provide only the saved `agent_0` model, retain the original task-role count, and set `collaboration_mode="centralized"` again. Preference trainers start their preference collection and reward-learning stages according to the new run's configuration.

## Restoring Value Heads

IAC and MAAC save value heads separately as `value_head.pt`. Loading a critic or shared actor directory through the constructor restores the language-model backbone and creates a new value head. Load the saved head explicitly before calling `train()` to retain its learned values.

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

For IAC, use the same `load_state_dict` step on `continued_iac.agents[i].value_head` when `use_separate_critic=False`, or on `continued_iac.critics[i].value_head` when it is `True`. Load each head from the corresponding path in the saving table; load separate critic backbones through `critic_model` or `critics` as well.

The value-head architecture must match the saved run. For IAC, reuse `IACConfig.value_head_hidden_dim` for shared heads and `IACConfig.critic_value_head_hidden_dim` for separate heads. For a custom MAAC head, pass the original `model_config={"critic_value_head_hidden_dim": ...}` when constructing the trainer. Keep the original `critic_type` (`"v"` or `"q"`) when continuing the same value-learning setup.

## Training State

{{% hint warning %}}
Saved weights support further training, but CoMLRL currently has no complete `resume_from_checkpoint` implementation. `save_model` does not save or restore optimizer state, scheduler state, epoch or environment-step counters, random-number-generator state, or rollout buffers. A new trainer creates fresh optimizers and starts its counters again.
{{% /hint %}}

When reference KL is enabled, a new trainer builds its frozen reference models from the actor sources supplied to that constructor. Reloading trained actors therefore uses those reloaded weights as the new reference, rather than restoring the previous run's reference models.

Iterative MAPL's `policy_checkpoint_dir` stores actor snapshots for comparator history. Its persisted preference pairs and replay files are also separate from `save_model`; creating a new iterative trainer does not automatically reconstruct the previous iteration or replay state from those files. See [Comparator]({{< relref "/docs/mapl/comparator" >}}) and [Replay Buffer]({{< relref "/docs/mapl/replay-buffer" >}}) for their roles in an active run.
