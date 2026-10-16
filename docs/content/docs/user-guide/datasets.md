---
title: Dataset Loading
linkTitle: Dataset Loading
weight: 3
aliases:
  - /docs/user-guide/datasets-evaluation-logging/
---

Users can keep task inputs and metadata in the same dataset, with a separate split for evaluation. This guide prepares training and evaluation data for the length-ratio task from the [quick start]({{< relref "docs/examples/comlrl-quick-start" >}}).

## Dataset Splits

Each dataset row is a dictionary. The default formatter reads its `prompt` field; additional fields can hold targets, labels, test cases, or other information required by the reward function. Trainers pass the original row to the reward function through `batch_items`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": [
        "Describe a city in the clouds.",
        "Invent a new holiday.",
        "Write a bedtime story for a dragon.",
        "Explain how teleportation might work.",
        "Tell a story about dinosaurs.",
        "Describe a world without electricity.",
        "Create a superhero with a unique power.",
        "Write a scene where the moon talks.",
        "Invent a new type of fruit.",
        "Design a playground on Mars.",
    ],
    "target_min": [2.0] * 10,
    "target_max": [3.0] * 10,
})

split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
```

Here, eight rows are used for training and two are held out for evaluation. `seed` makes the split repeatable; it does not set the random seed for model generation. Existing training and validation splits can be used directly.

The current training data loaders preserve dataset order. Users can shuffle the data with `train_dataset.shuffle(seed=42)` before creating the trainer. This produces a fixed order, without reshuffling each epoch.

## Prompts and Rewards

Users can share a prompt format across agents with one formatter, or assign different roles with a list of formatters in actor order. Only fields used by the formatter enter the model prompt; the reward function can still read the full row.

```python
import math

def brief_formatter(item):
    return f"Write a short response to this task:\n{item['prompt']}"

def detailed_formatter(item):
    return f"Write a detailed response to this task:\n{item['prompt']}"

formatters = [brief_formatter, detailed_formatter]

def reward_func(brief_completions, detailed_completions, *, batch_items):
    item = batch_items[0]
    rewards = []
    for brief, detailed in zip(brief_completions, detailed_completions):
        if not brief:
            rewards.append(0.0)
            continue
        ratio = len(detailed) / len(brief)
        distance = max(item["target_min"] - ratio, ratio - item["target_max"], 0.0)
        rewards.append(math.exp(-distance))
    return rewards
```

Each completion argument is a list of strings for one agent. The trainers used here call the reward function for one dataset row at a time, with that row available as `batch_items[0]`. This function returns one shared reward per aligned pair of responses. The task rewards a character-count ratio; it does not measure writing quality, and the agents generate independently.

Users can check the data and reward function without loading models:

```python
item = train_dataset[0]
assert item["prompt"] in brief_formatter(item)
assert reward_func(["Hi"], ["Hello"], batch_items=[item]) == [1.0]
assert reward_func([""], ["Hello"], batch_items=[item]) == [0.0]
```

For multi-turn tasks, formatters may also accept `external_prompts`. The transition interface is described in [Multi-Turn Interaction]({{< relref "docs/multi-turn/transition" >}}).

## Training and Evaluation

Users can provide training and evaluation data separately through `train_dataset` and `eval_dataset`. Evaluation uses the training formatters and reward function without updating model weights. Outputs are sampled with the configured generation settings, so repeated evaluations can produce different scores.

```python
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer

config = MAGRPOConfig(
    num_agents=2,
    num_turns=1,
    num_train_epochs=1,
    num_generations=4,
    max_new_tokens=128,
    eval_interval=4,
    eval_num_samples=len(eval_dataset),
    eval_batch_size=1,
    logging_steps=4,
)

trainer = MAGRPOTrainer(
    agent_model="Qwen/Qwen2.5-0.5B",
    num_agents=2,
    formatters=formatters,
    reward_func=reward_func,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
)

trainer.train()
metrics = trainer.evaluate(num_eval_samples=len(eval_dataset))
print(metrics)
```

For the MAREINFORCE family, IAC, and MAAC, `eval_interval` counts training dataset items. Evaluation runs before item zero of each epoch, then at the configured interval; `0` enables manual evaluation only. `eval_num_samples` specifies a positive limit on evaluated rows, starting at the beginning of the evaluation split. Setting it to the split length evaluates all rows. `eval_batch_size` controls the evaluation data loader batch size, not the number of responses generated per agent.

MAGRPO, IAC, and MAAC generate one response per agent per turn during evaluation. IAC and MAAC use their rollout collection path, with the configured turn structure, but their `evaluate()` method takes no sample-count argument:

```python
# For an existing IACTrainer or MAACTrainer:
ac_trainer.args.eval_num_samples = len(eval_dataset)
metrics = ac_trainer.evaluate()
```

Without an `eval_dataset`, `evaluate()` returns an empty dictionary. For MAPL, manual `evaluate(num_eval_samples=...)` is also available; the automatic evaluation schedule follows the preference-update or actor-training stage rather than the dataset-item schedule above.


[Logging Metric]({{< relref "docs/user-guide/logging" >}}) covers saving evaluation results, recording training curves, and adding custom metrics.
