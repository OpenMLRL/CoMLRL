---
title: Training Parallelization
linkTitle: Training Parallelization
weight: 4
---

CoMLRL trainers run in a Python process. For IAC, `parallel_training="none"` is
the default; `"mp"` enables scheduling agents and critics on explicitly assigned
devices. Neither mode requires a job scheduler.

## Model Parallelization

Pass an `IACConfig` to your `IACTrainer` through its `args` parameter. For example,
with four visible CUDA devices and two agents:

```python
from comlrl.trainers.actor_critic import IACConfig

config = IACConfig(
    num_agents=2,
    num_turns=1,
    use_separate_critic=True,
    parallel_training="mp",
    agent_devices=["cuda:0", "cuda:1"],
    critic_devices=["cuda:2", "cuda:3"],
)
```

`agent_devices` is required in `"mp"` mode. `critic_devices` is also required
when `use_separate_critic=True`. Each model stays on its assigned device; this
does not split an individual model across GPUs. Responses are passed to the
reward function and rewards are used for each agent's update.

Device indices refer to the GPUs visible to the Python process. Configure the
device lists to match your machine and the memory needed by each model. Changing
device placement or parallel scheduling can change numerical training results.
