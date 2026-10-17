---
title: Training Parallelization
linkTitle: Training Parallelization
weight: 6
---

CoMLRL can place different models on different GPUs. With `parallel_training="mp"`, each agent runs on its own device from `agent_devices`, and the agents generate concurrently; critics, reward models, comparators, and reference models take their own device settings. The default, `"none"`, keeps the agents on one device. Both modes run in one Python process and require no job scheduler.

Each model stays on its assigned device; a single model is not split across GPUs. Device indices refer to the GPUs visible to the Python process. A setting for several models takes either one device per model or one device for all of them, so users can match the placement to the available GPUs and each model's memory. Device placement and parallel scheduling can affect numerical training results.

## Device Placement

{{% hint info "**Device Placement**" %}}
**Device Placement** parameters:

- `agent_devices`: Agent devices; required in `"mp"` mode
- `critic_devices`: One device for a centralized critic, or one per agent for separate critics; required in `"mp"` mode for MAAC and separate-critic IAC, while MARLHF defaults to the agent devices
- `reward_model_device`: MARLHF reward-model device; defaults to `"cuda:0"`, or `None` for the trainer device
- `comparator_devices`: Devices for local comparator models in iterative PbRL; defaults to the agent devices
- `reference_devices`: Devices for frozen reference models with reference KL; defaults to the agent devices
{{% /hint %}}

## Example

The following configuration puts each model of an iterative MARLHF run on its own GPU: two agents, a centralized critic for MAAC policy learning, the reward model, and copies of the current agents as comparators.

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
