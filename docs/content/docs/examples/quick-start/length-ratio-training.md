---
title: Length-Ratio Training
weight: 1
aliases:
  - /docs/examples/comlrl-quick-start/
---

This tutorial demonstrates a simple two-agent training task with a length-ratio reward. Both agents receive the same prompt and generate independently; the second agent does not see the first agent's response. The reward encourages the second output to contain 2–3× as many characters as the first. It does not evaluate story quality or semantic collaboration.

Users can run the example from an installed source checkout:

```bash
python examples/story-len-ratio.py
```

Memory requirements depend on the model, number of agents, batch size, and response length. W&B logging is disabled by default. The `--wandb` flag enables it; `--wandb-project` and `--wandb-entity` are optional.

## Import Libraries

```python
import math
from functools import partial
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer
```

## Dataset Preparation

The dataset contains creative prompts for both agents.

```python
train_data = {
    "prompt": [
        "Describe a city in the clouds:",
        "Invent a new holiday and explain it:",
        "Write a bedtime story for a dragon:",
        "Explain how teleportation might work:",
        "Tell a joke about dinosaurs:",
        "Describe a world without electricity:",
        "Create a superhero with a unique power:",
        "Write a scene where the moon talks:",
        "Invent a new type of fruit:",
        "Design a playground on Mars:",
    ]
}
train_dataset = Dataset.from_dict(train_data)
```

## Agent Initialization

The tokenizer converts text into model tokens. Each agent uses a separate model instance.

```python
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
agents = [AutoModelForCausalLM.from_pretrained(model_name) for _ in range(2)]
```

## Define the Reward Function

The reward function evaluates the ratio of the agents' output lengths. It returns the maximum reward (1.0) when the second output contains 2–3× as many characters as the first. Outside this range, the reward decays exponentially with the distance from the nearest boundary.

```python
def proper_length_ratio_reward(
    completions1, completions2, target_min=2.0, target_max=3.0
):
    rewards = []
    for c1, c2 in zip(completions1, completions2):
        len1, len2 = len(c1), len(c2)

        if len1 == 0:
            rewards.append(0.0)
            continue

        ratio = len2 / len1

        if target_min <= ratio <= target_max:
            reward = 1.0
        else:
            if ratio < target_min:
                distance = target_min - ratio
            else:
                distance = ratio - target_max

            reward = math.exp(-distance)

        rewards.append(float(reward))

    return rewards
```

## Configure Training

Users can configure training through `MAGRPOConfig`, including the learning rate, batch size, and number of generations per agent per prompt.

```python
output_dir = "./magrpo_multi_reward_output"
config = MAGRPOConfig(
    num_train_epochs=3,
    agent_learning_rate=5e-5,
    logging_steps=10,
    num_generations=8,
    num_turns=1,
    max_new_tokens=128,
)
```

## Create the Trainer

The following example initializes `MAGRPOTrainer` with the agents, reward function, and configuration defined above. `RewardProcessors.scale` multiplies task rewards by 100. With the default `advantage_normalization=True`, normalization largely cancels this positive scaling, so it does not imply a 100× stronger policy gradient.

```python
configured_reward_func = partial(
    proper_length_ratio_reward, target_min=2, target_max=3
)

trainer = MAGRPOTrainer(
    agents=agents,
    reward_func=configured_reward_func,
    reward_processor=RewardProcessors.scale(factor=100.0),
    args=config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
```

## Run Training

The following code trains both agents to maximize the shared reward and saves the trained models.

```python
trainer.train()
trainer.save_model(f"{output_dir}/models")
```
