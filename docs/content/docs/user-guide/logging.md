---
title: Logging Metric
linkTitle: Logging Metric
weight: 4
---

Save evaluation results locally, record training curves in Weights & Biases, and add task-specific metrics. The examples below build on the trainer and held-out split in [Dataset Loading]({{< relref "docs/user-guide/datasets" >}}).

## Local Metrics

Evaluation returns a metrics dictionary even when W&B is disabled. Save it locally if you only need a record of a completed evaluation:

```python
import json
from pathlib import Path

# Use the MAGRPO trainer and eval_dataset from the Dataset Loading guide.
metrics = trainer.evaluate(num_eval_samples=len(eval_dataset))
Path("eval-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
```

## Weights & Biases

To record training curves in Weights & Biases, pass `wandb_config` when constructing the trainer:

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

Authenticate W&B in your environment before enabling it. Omit `wandb_config` or set it to `None` to disable W&B. The supported run settings include `project`, `entity`, `name` (or `run_name`), `tags`, and `dir`; `config_sections` adds your dataset and experiment settings to the recorded configuration. `config_sections.output.verbose` controls trainer progress output.

## Training Metrics

For the MARL trainers, logs use `env_step`, which counts joint-action reward evaluations. It is different from a dataset index or optimizer-update count, especially with multiple generations or turns. `logging_steps` sets the minimum gap in this counter between standard training metric logs, emitted when rollout buffers are processed. Evaluation follows `eval_interval` independently.

| Metric | Meaning |
| --- | --- |
| `turn_1/reward_mean` | Mean immediate reward in the logged training rollouts at turn 1 |
| `turn_1/expected_return` | Mean return stored for those training rollouts |
| `eval/turn_1/reward_mean` | Mean immediate reward in evaluation rollouts at turn 1 |
| `eval/turn_1/expected_return` | Mean evaluation return from turn 1 |

Turn numbers start at 1. MAGRPO evaluation returns are undiscounted sums of rewards from the current turn onward; IAC and MAAC report their collected rollout returns. Rewards reflect the configured reward processing, and actor-critic returns can also include reference-KL shaping when enabled. The single-turn example in [Dataset Loading]({{< relref "docs/user-guide/datasets" >}}) uses the default identity reward processor and no reference KL, so its reward and return metrics describe the same task score.

## Custom Metrics

MAGRPO accepts an `eval_logger` and an `eval_aggregator` together. The logger receives completions in `[agent][dataset item][turn]` order plus the original `batch_items` and raw `prompts`. The aggregator returns scalar metrics without the `eval/` prefix; the trainer adds it.

For example, measure the average response length across agents at each turn:

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

The resulting dictionary includes `eval/turn_1/response_chars`. These callbacks also receive metadata fields, so you can compute task-specific accuracy or group metrics without placing reference answers in the prompts.

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

This callback runs during both training and evaluation. Evaluation averages its values across evaluated dataset items and adds `eval/`; training logs its values at rollout collection time, separately from the standard `logging_steps` gate.
