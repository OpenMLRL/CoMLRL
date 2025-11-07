---
title: CoMLRL Quick Demo
weight: 1
---

This example demonstrates MAGRPO on a small storytelling task with a length‑ratio reward. Two agents generate per prompt: Agent 1 produces a compact setup; Agent 2 produces a longer continuation. The reward favors a 2–3× length ratio between Agent 2 and Agent 1.

## Imports

What’s used: Hugging Face models/tokenizer, Hugging Face `Dataset` to hold prompts, MAGRPO trainer/config, a simple reward processor (for scaling), and utilities.

```python
import math
from functools import partial

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer
```

## Reward

We want Agent 2’s output to be roughly 2–3× longer than Agent 1’s. Outside that band, we decay the reward exponentially by the distance from the nearest bound.

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

## Data (prompts)

A tiny prompt set is enough for a functional demo; expand this for real experiments.

```python
train_data = {
    "prompt": [
        "Write a story about a robot:",
        "Explain quantum physics:",
        "Create a recipe for chocolate cake:",
        "Describe a city in the clouds:",
        "Invent a new holiday and explain it:",
        "Write a bedtime story for a dragon:",
        "Explain how teleportation might work:",
        "Design a futuristic bicycle:",
        "Tell a joke about dinosaurs:",
        "Write a poem about the ocean at night:",
        "Describe a world without electricity:",
        "Create a superhero with a unique power:",
        "Write a scene where the moon talks:",
        "Explain black holes to a 5-year-old:",
        "Invent a new type of fruit:",
        "Design a playground on Mars:",
        "Write a love letter between two stars:",
        "Invent a game played by aliens:",
        "Explain Wi-Fi to someone from the 1800s:",
        "Create a workout plan for robots:",
        "Describe a hotel at the bottom of the ocean:",
        "Write a story about a lost shadow:",
        "Invent a musical instrument from glass:",
        "Design a zoo for extinct animals:",
        "Write a diary entry from a raindrop:",
        "Describe a world where pets can talk:",
        "Explain how dreams are made:",
        "Create a menu for a restaurant in space:",
        "Write a letter from a tree to a human:",
        "Describe a rainbow factory:",
        "Write a scene from a robot cooking show:",
        "Explain the weather like a pirate would:",
    ]
}
train_dataset = Dataset.from_dict(train_data)
```

## Tokenizer

Load the tokenizer from your chosen base model. For quick tests, a smaller model is fine.

```python
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## Models

Two identical base models act as two agents. Scale model size to match VRAM; the original script notes ~24GB for larger runs.

```python
agents = [AutoModelForCausalLM.from_pretrained(model_name) for _ in range(2)]
```

## MAGRPO config

Use a modest configuration for a fast run; increase epochs/generations for stronger results.

```python
config = MAGRPOConfig(
    output_dir="./magrpo_multi_reward_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    num_generations=8,
    max_new_tokens=128,
)
```

## Trainer setup

Scale rewards for a smoother learning signal; keep W&B optional.

```python
wandb_config = {
    "project": "mlrl",
    "entity": "OpenMLRL",
    "name": "qwen-magrpo-length-ratio",
}

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
    wandb_config=wandb_config,
)
```

## Train and save

Run a short training loop and checkpoint the final models.

```python
trainer.train()
trainer.save_model(f"{config.output_dir}/final_models")
print("Training complete!")
```

---

Notes

- Based on `examples/story-len-ratio.py`.
- VRAM tip: the original script notes ~24GB VRAM for larger models; use a smaller model if constrained.
- For longer runs, increase `num_train_epochs`, adjust `num_generations`, and expand the prompt list.
- See the User Guide → Multi‑Turn for Joint Mode, External feedback, and Termination settings.
