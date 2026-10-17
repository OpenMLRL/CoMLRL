---
title: Training Parallelization
linkTitle: Training Parallelization
weight: 6
tocDepth: 3
---

CoMLRL trainers run in a Python process. For IAC and the preference trainers,
`parallel_training="none"` is the default; `"mp"` enables scheduling agents and
other training entities on explicitly assigned devices. Neither mode requires a
job scheduler.

## Model Parallelization

Each model stays on its assigned device; this does not split an individual model
across GPUs. Device indices refer to the GPUs visible to the Python process.
Users can choose device assignments based on the available devices and each
model's memory requirements. Device placement and parallel scheduling can affect
numerical training results.

### Actor-Critic

Users can configure `IACTrainer` by passing an `IACConfig` through its `args` parameter. The following
configuration uses four visible CUDA devices and two agents:

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
when `use_separate_critic=True`. Responses are passed to the reward function,
and the rewards are used for each agent's update.

### PbRL

Preference trainers place the reward model and comparator through their own
settings. The following iterative MARLHF configuration uses six visible CUDA
devices, two agents, and MAAC for the online RL stage:

```python
from comlrl.trainers.preference import MARLHFIterConfig

config = MARLHFIterConfig(
    num_agents=2,
    num_turns=1,
    rl_algorithm="maac",
    parallel_training="mp",
    agent_devices=["cuda:0", "cuda:1"],
    critic_devices="cuda:2",
    reward_model_device="cuda:3",
    comparator_policy="current_copy",
    comparator_devices=["cuda:4", "cuda:5"],
)
```

The shared MAAC critic takes one device and defaults to the first agent device;
with `rl_algorithm="iac"` and separate critics, `critic_devices` takes one device
per agent. The joint reward model trains and scores on `reward_model_device`.
`current_copy` copies the current actors to `comparator_devices`, one device per
role, for each preference-collection phase.
