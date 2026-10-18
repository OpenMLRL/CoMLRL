---
title: Multi-Turn Feedback
weight: 2
---

In this example, two agents answer a task, receive feedback on their answers, and then answer again.

## Setup

The task asks the second agent to write a response 2 to 3 times as long as the first agent's response.

```python
import math

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer

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

## Feedback Between Turns

After the first turn, this function writes a new prompt for each agent. The prompt reports how long the answers were.

```python
def length_feedback(prompt, agent_completions, num_agents, **history):
    lengths = [len(text) for text in agent_completions]
    ratio = lengths[1] / max(lengths[0], 1)
    return [
        f"{prompt}\n\nYour last answer had {lengths[i]} characters. "
        f"The second answer was {ratio:.1f} times as long as the first; "
        "the goal is 2 to 3 times. Please write an improved answer."
        for i in range(num_agents)
    ]
```

## Two-Turn Training

`num_turns=2` adds the second turn, and `external_transition` connects the feedback function.

```python
config = MAGRPOConfig(
    num_turns=2,
    num_train_epochs=2,
    num_generations=4,
    max_new_tokens=64,
    logging_steps=1,
    early_termination_threshold=0.9,
)
trainer = MAGRPOTrainer(
    agents=[AutoModelForCausalLM.from_pretrained(model_name) for _ in range(2)],
    tokenizer=tokenizer,
    reward_func=length_ratio_reward,
    external_transition=length_feedback,
    train_dataset=train_dataset,
    args=config,
)
trainer.train()
```

`early_termination_threshold=0.9` ends a branch after the first turn only when its answers already reach the goal. The default, −0.2, would end every branch after the first turn, because this reward is never negative.

## What Happens in Each Step

- In the first turn, each agent writes 4 answers, which form 4 joint answers.
- The feedback function turns each joint answer into new prompts.
- In the second turn, each agent answers again, so one task has up to 16 joint answers.
- A first-turn answer is scored by its own reward plus the average reward of the second-turn answers that follow it.
- With W&B enabled, `turn_1/reward_mean` and `turn_2/reward_mean` show whether the second answers are better.
