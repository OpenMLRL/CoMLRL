---
title: Dataset Loading
linkTitle: Dataset Loading
weight: 3
aliases:
  - /docs/user-guide/datasets-evaluation-logging/
---

Keep task inputs and metadata together in a dataset and reserve a separate evaluation split. This guide uses the length-ratio task from the [quick start]({{< relref "docs/examples/comlrl-quick-start" >}}) to prepare data for training and evaluation.

## Dataset Splits

Each dataset row is a dictionary. The default formatter reads its `prompt` field; additional fields can hold targets, labels, test cases, or other information needed by your reward function. Trainers pass the original row to the reward function through `batch_items`.

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

Here, eight rows are used for training and two are held out for evaluation. `seed` makes the split repeatable; it does not set the random seed for model generation. If your dataset already provides separate training and validation splits, pass those directly instead.

The current training data loaders preserve dataset order. Use `train_dataset.shuffle(seed=42)` before creating the trainer if you want a shuffled order; this produces a fixed order rather than a new shuffle each epoch.

## Prompts and Rewards

Pass one formatter to share a prompt format across agents, or a list of formatters in actor order to assign different roles. Only fields used by the formatter enter the model prompt; the reward function can still read the full row.

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

You can check the data and reward without loading any models:

```python
item = train_dataset[0]
assert item["prompt"] in brief_formatter(item)
assert reward_func(["Hi"], ["Hello"], batch_items=[item]) == [1.0]
assert reward_func([""], ["Hello"], batch_items=[item]) == [0.0]
```

For multi-turn tasks, formatters may also accept `external_prompts`. See [Multi-Turn Interaction]({{< relref "docs/multi-turn/transition" >}}) for the transition interface.

## Training and Evaluation

Supply `train_dataset` and `eval_dataset` separately. Evaluation uses the same formatters and reward function as training, without updating model weights. It still samples model outputs using the generation settings, so repeated evaluations can produce different scores.

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

For the MAREINFORCE family, IAC, and MAAC, `eval_interval` counts training dataset items: evaluation runs before item zero of each epoch, then at every configured interval. Set it to `0` for manual evaluation only. `eval_num_samples` limits the number of dataset rows evaluated, starting from the beginning of the evaluation split. Use a positive count; to evaluate the entire split, use its length. `eval_batch_size` controls the evaluation data loader batch size, not the number of generated responses per agent.

MAGRPO, IAC, and MAAC generate one response per agent per turn during evaluation. IAC and MAAC use their rollout collection path, with the configured turn structure, but their `evaluate()` method takes no sample-count argument:

```python
# For an existing IACTrainer or MAACTrainer:
ac_trainer.args.eval_num_samples = len(eval_dataset)
metrics = ac_trainer.evaluate()
```

Without an `eval_dataset`, `evaluate()` returns an empty dictionary. For MAPL, manual `evaluate(num_eval_samples=...)` is also available; the automatic evaluation schedule follows the preference-update or actor-training stage rather than the dataset-item schedule above.


See [Logging Metric]({{< relref "docs/user-guide/logging" >}}) for saving evaluation results, recording training curves, and adding custom metrics.
