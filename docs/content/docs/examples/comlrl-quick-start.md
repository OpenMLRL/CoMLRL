---
title: CoMLRL Quick Start
weight: 1
---

This tutorial demonstrates a simple two-agent training task with a length-ratio reward. Both agents receive the same prompt and generate independently; the second agent does not see the first agent's response. The reward encourages the second output to contain 2–3× as many characters as the first. It does not evaluate story quality or semantic collaboration.

Run the example directly from an installed source checkout:

```bash
python examples/story-len-ratio.py
```

Memory requirements depend on the model, number of agents, batch size, and response length. W&B logging is disabled by default. To enable it, add `--wandb` (and optionally `--wandb-project` or `--wandb-entity`) to the command above.

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

We first create a dataset of creative prompts for the agents to work on.

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

We load a tokenizer to convert text into tokens that the model can process and initialize two separate instances.

```python
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
agents = [AutoModelForCausalLM.from_pretrained(model_name) for _ in range(2)]
```

## Define the Reward Function

The reward function measures how well the agents collaborate. It gives maximum reward (1.0) when the second agent's output is 2–3× longer than the first agent's. If the length ratio falls outside this range, the reward decays exponentially based on how far it deviates.

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

We set up the training configuration with hyperparameters like learning rate, batch size, and the number of generations each agent produces per prompt.

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

We instantiate the MAGRPO trainer with our agents, reward function, and configuration. `RewardProcessors.scale` multiplies task rewards by 100. With the default `advantage_normalization=True`, this positive scaling is largely canceled by normalization and should not be interpreted as a 100× stronger policy gradient.

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

Finally, we start the training process. The trainer will optimize both agents to maximize the collaborative reward, then save the trained models.

```python
trainer.train()
trainer.save_model(f"{output_dir}/models")
```
