---
title: "Shared Actor-Critic"
weight: 2
---

This case study shows how to run IAC with a shared critic (value heads attached to the actors). It reduces memory use and keeps training simple for small models.

## When to Use Shared Actor-Critic

- You want the lowest VRAM footprint for actor-critic.
- Your task is single-turn or short-horizon.
- You are fine with slightly noisier value estimates compared to a separate critic.

## Key Settings

- Set `iac.use_separate_critic` to `false`.
- Do not provide `critic_model` or `critics`.
- Keep `num_turns=1` unless you also provide an `external_transition`.

## Minimal Example (Python)

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from comlrl.trainers.actor_critic import IACConfig, IACTrainer

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("trl-lib/tldr", split="train").select(range(128))

def length_balance_reward(agent1, agent2):
    a = agent1[0]
    b = agent2[0]
    denom = max(len(a), len(b), 1)
    return [1.0 - abs(len(a) - len(b)) / denom]

trainer = IACTrainer(
    agent_model=model_name,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_func=length_balance_reward,
    formatters=[lambda ex: ex["prompt"]] * 2,
    args=IACConfig(
        num_agents=2,
        num_turns=1,
        use_separate_critic=False,
        rollout_buffer_size=4,
        train_batch_size=4,
    ),
)
trainer.train()
```

## Notes

- If you pass `agents`, set `agent_model.name` to `null` in your config to avoid conflicts.
- Shared-critic mode does not accept `critic_model` or `critics`.
