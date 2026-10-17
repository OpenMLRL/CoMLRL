---
title: Training Initialization
linkTitle: Training Initialization
weight: 3
tocDepth: 3
aliases:
  - /docs/user-guide/datasets/
  - /docs/user-guide/datasets-evaluation-logging/
  - /docs/user-guide/logging/
---

Training is initialized from a task set, a class of tasks that share an input format and a reward. Formatters turn each task into agent prompts, and a reward function scores the agents' joint responses. The examples use the length-ratio task from the [quick start]({{< relref "docs/examples/comlrl-quick-start" >}}).

## Task Sets

A task set is a Hugging Face `Dataset` or `IterableDataset` whose rows are task items, each a dictionary. The default formatter reads the `prompt` field; additional fields can hold targets, labels, test cases, or other information required by the reward function. Trainers pass the original task item to the reward function through `batch_items`.

```python
from datasets import Dataset

tasks = Dataset.from_dict({
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

split = tasks.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
```

Here, eight task items are used for training and two are held out for evaluation. `seed` makes the split repeatable; it does not set the random seed for model generation. Existing training and evaluation splits can be used directly.

The current training data loaders preserve task order. Users can shuffle the task items with `train_dataset.shuffle(seed=42)` before creating the trainer. This produces a fixed order, without reshuffling each epoch.

## Prompts from Tasks

Users can share a prompt format across agents with one formatter, or assign different roles with a list of formatters in actor order. Only fields used by a formatter enter the model prompt; the reward function can still read the full task item.

```python
def brief_formatter(item):
    return f"Write a short response to this task:\n{item['prompt']}"

def detailed_formatter(item):
    return f"Write a detailed response to this task:\n{item['prompt']}"

formatters = [brief_formatter, detailed_formatter]
```

For multi-turn tasks, formatters may also accept `external_prompts`. The transition interface is described in [Multi-Turn Interaction]({{< relref "docs/multi-turn/transition" >}}).

## Rewards

Both rule-based rewards and LLM reward models are passed to trainers through `reward_func`. The reward function receives one list of completions per agent, in actor order. If it accepts a `batch_items` keyword, trainers also pass the task items. It returns one reward per joint response. The trainers used here call it for one task item at a time, with that item available as `batch_items[0]`.

`reward_processor` transforms each returned reward before training; for example, `RewardProcessors.scale(factor=10.0)` from `comlrl.utils.reward_processor` multiplies rewards by 10. By default, rewards are unchanged.

### Rule-Based Rewards

A rule-based reward computes scores directly from the responses and task items, for example with task checks, verifiers, or test harnesses. The following function scores the length-ratio task:

```python
import math

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

This function returns one shared reward per aligned pair of responses. The task rewards a character-count ratio; it does not measure writing quality, and the agents generate independently.

Users can check the task set, formatters, and reward function without loading models:

```python
item = train_dataset[0]
assert item["prompt"] in brief_formatter(item)
assert reward_func(["Hi"], ["Hello"], batch_items=[item]) == [1.0]
assert reward_func([""], ["Hello"], batch_items=[item]) == [0.0]
```

### LLM-Based Reward Models

A reward function can also score responses with an existing LLM reward model; trainers use only the values it returns. The following sketch wraps a reward model with a single-score classification head; `reward_model_id` is a placeholder:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

reward_model_id = "your-org/your-reward-model"
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id).eval()

@torch.no_grad()
def model_reward(brief_completions, detailed_completions, *, batch_items):
    task = batch_items[0]["prompt"]
    texts = [
        f"{task}\n\nBrief: {brief}\n\nDetailed: {detailed}"
        for brief, detailed in zip(brief_completions, detailed_completions)
    ]
    inputs = reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return reward_model(**inputs).logits[:, 0].tolist()

# Pass it when constructing a trainer:
# reward_func=model_reward,
```

Users can load the reward model once, outside the function, and place it on a device with free memory; trainers do not move or update it. To learn a reward model from preferences instead, see [LLM Reward Models]({{< relref "/docs/user-guide/training-entities#llm-reward-models" >}}).

## Training and Evaluation

Users can provide training and evaluation task items separately through `train_dataset` and `eval_dataset`. Evaluation uses the training formatters and reward function without updating model weights. Outputs are sampled with the configured generation settings, so repeated evaluations can produce different scores.

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

For the MAREINFORCE family, IAC, and MAAC, `eval_interval` counts training task items. Evaluation runs before item zero of each epoch, then at the configured interval; `0` enables manual evaluation only. `eval_num_samples` specifies a positive limit on evaluated task items, starting at the beginning of the evaluation split. Setting it to the split length evaluates all task items. `eval_batch_size` controls the evaluation data loader batch size, not the number of responses generated per agent.

MAGRPO, IAC, and MAAC generate one response per agent per turn during evaluation. IAC and MAAC use their rollout collection path, with the configured turn structure, but their `evaluate()` method takes no sample-count argument:

```python
# For an existing IACTrainer or MAACTrainer:
ac_trainer.args.eval_num_samples = len(eval_dataset)
metrics = ac_trainer.evaluate()
```

Without an `eval_dataset`, `evaluate()` returns an empty dictionary. For PbRL trainers, manual `evaluate(num_eval_samples=...)` is also available; the automatic evaluation schedule follows the preference-update or actor-training stage rather than the task-item schedule above.

## Logging

CoMLRL supports local evaluation records, training curves in Weights & Biases, and task-specific metrics. The examples use the trainer and evaluation split above.

### Local Metrics

Evaluation returns a metrics dictionary even when W&B is disabled. Users can save this dictionary locally to record a completed evaluation:

```python
import json
from pathlib import Path

metrics = trainer.evaluate(num_eval_samples=len(eval_dataset))
Path("eval-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
```

### Weights & Biases

Users can record training curves in Weights & Biases by supplying `wandb_config` when creating the trainer:

```python
wandb_config = {
    "project": "comlrl",
    "name": "length-ratio-magrpo",
    "tags": ["length-ratio", "magrpo"],
    "config_sections": {
        "dataset": {"name": "length-ratio", "split_seed": 42},
        "output": {"verbose": True},
    },
}

# Add this keyword when constructing MAGRPOTrainer:
# wandb_config=wandb_config,
```

Users can authenticate W&B in their environment before enabling logging. Omitting `wandb_config` or setting it to `None` disables W&B. Supported run settings include `project`, `entity`, `name` (or `run_name`), `tags`, and `dir`. The `config_sections` field adds task and experiment settings to the recorded configuration; `config_sections.output.verbose` controls trainer progress output.

### Training Metrics

Trainers log the following metrics by default, once per turn `k` (starting from 1). [Custom metrics](#custom-metrics) can add task-specific values alongside them.

| Metric | Meaning |
| --- | --- |
| `turn_k/reward_mean` | Mean reward of the training rollouts at turn `k` |
| `turn_k/expected_return` | Mean return of the training rollouts from turn `k` |
| `eval/turn_k/reward_mean` | Mean reward of the evaluation rollouts at turn `k` |
| `eval/turn_k/expected_return` | Mean return of the evaluation rollouts from turn `k` |

With reference KL enabled, `reference_kl_mean` and `reference_kl_penalty_mean` are also logged for each turn. Training metrics are logged against `env_step`, the number of joint-action reward evaluations, which grows faster than the task index with multiple generations or turns. `logging_steps` sets the minimum `env_step` gap between training logs; evaluation metrics follow `eval_interval`.

### Custom Metrics

Users can add custom evaluation metrics to MAGRPO by supplying an `eval_logger` and an `eval_aggregator` together. The logger receives completions in `[agent][task item][turn]` order plus the original `batch_items` and raw `prompts`. The aggregator returns scalar metrics without the `eval/` prefix; the trainer adds it.

The following example measures the average response length across agents at each turn:

```python
def eval_logger(*, agent_completions_turns, batch_items, prompts):
    return [
        {
            f"turn_{turn + 1}/response_chars": sum(
                len(agent[row][turn]) for agent in agent_completions_turns
            ) / len(agent_completions_turns)
            for turn in range(len(agent_completions_turns[0][row]))
        }
        for row in range(len(batch_items))
    ]

def eval_aggregator(rows, *, num_turns):
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for turn in range(num_turns)
        for key in [f"turn_{turn + 1}/response_chars"]
    } if rows else {}

# Add both keywords when constructing MAGRPOTrainer:
# eval_logger=eval_logger,
# eval_aggregator=eval_aggregator,
```

The resulting dictionary includes `eval/turn_1/response_chars`. These callbacks also receive task-item fields, allowing task-specific accuracy or group metrics to be computed without reference answers in the prompts.

IAC and MAAC instead accept `metrics_callback(rollouts)`, which returns a dictionary of numeric values. Each rollout sample provides fields such as `agent_idx`, `prompt`, `completion`, and `metadata`:

```python
def metrics_callback(rollouts):
    if not rollouts:
        return {}
    return {
        "response_chars": sum(len(sample.completion) for sample in rollouts)
        / len(rollouts)
    }

# Add metrics_callback=metrics_callback to IACTrainer(...) or MAACTrainer(...).
```

This callback runs during both training and evaluation. Evaluation averages its values across evaluated task items and adds `eval/`; training logs its values at rollout collection time, separately from the standard `logging_steps` gate.
