---
title: Resuming Training
weight: 1
---

This example trains two small agents for two epochs, then continues the same run for two more epochs from a checkpoint. Both parts are logged to one W&B run.

## Setup

The task asks the second agent to write a response 2 to 3 times as long as the first agent's response.

```python
import math
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer
from comlrl.utils import read_checkpoint_metadata

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = Dataset.from_dict({
    "prompt": [
        "Describe a city in the clouds:",
        "Invent a new holiday and explain it:",
        "Write a bedtime story for a dragon:",
        "Design a playground on Mars:",
    ]
})


def length_ratio_reward(completions1, completions2):
    rewards = []
    for first, second in zip(completions1, completions2):
        ratio = len(second) / max(len(first), 1)
        distance = 0.0 if 2 <= ratio <= 3 else min(abs(ratio - 2), abs(ratio - 3))
        rewards.append(math.exp(-distance))
    return rewards
```

The same function builds the trainer for both parts of the run, so the two trainers match.

```python
def build_trainer(num_train_epochs, wandb_config):
    config = MAGRPOConfig(
        num_turns=1,
        num_train_epochs=num_train_epochs,
        num_generations=4,
        max_new_tokens=64,
        logging_steps=1,
    )
    return MAGRPOTrainer(
        agents=[AutoModelForCausalLM.from_pretrained(model_name) for _ in range(2)],
        tokenizer=tokenizer,
        reward_func=length_ratio_reward,
        train_dataset=train_dataset,
        args=config,
        wandb_config=wandb_config,
    )
```

## Training with Checkpoints

With `checkpoint_dir`, the trainer saves a checkpoint after every epoch.

```python
wandb_config = {"project": "comlrl-examples", "name": "resume-demo"}
checkpoint_dir = "./checkpoints/resume-demo"

trainer = build_trainer(num_train_epochs=2, wandb_config=wandb_config)
trainer.train(checkpoint_dir=checkpoint_dir, checkpoint_interval=1)
```

The directory now contains `checkpoint-1` and `checkpoint-2`.

## Continuing the Run

The newest checkpoint stores the W&B run ID. Adding it to `wandb_config` sends the new logs to the same run.

```python
checkpoint = max(
    Path(checkpoint_dir).glob("checkpoint-*"),
    key=lambda path: int(path.name.rsplit("-", 1)[1]),
)
identity = read_checkpoint_metadata(checkpoint)["wandb"]
if identity:
    wandb_config = {**wandb_config, **identity, "resume": "allow"}

trainer = build_trainer(num_train_epochs=4, wandb_config=wandb_config)
trainer.train(
    resume_from_checkpoint=str(checkpoint),
    checkpoint_dir=checkpoint_dir,
    checkpoint_interval=1,
)
```

`num_train_epochs` counts the whole run, so `4` adds two epochs to the two that are done. The same code also resumes a run that stopped in the middle.

## Why the Curves Line Up

- Metrics are logged at `env_step`, and every checkpoint saves `env_step` and the random states.
- The resumed run starts from the saved `env_step`, in the same W&B run.
- After an interruption, the work since the last checkpoint is done again. W&B skips steps that are not larger than the last logged one, so the curve has no duplicates.
- The models, data, reward function, and other settings have to stay the same. Only a few settings, such as `num_train_epochs`, devices, and logging intervals, may change.
