---
title: Training Parallelization
linkTitle: Training Parallelization
weight: 6
---

CoMLRL supports fine-tuning multi-LLM systems with larger models and more agents when multiple GPUs are available.
Currently, CoMLRL supports one training parallelization mode `mp` (model parallelization).

{{% hint success %}}
We will support more parallelization modes (e.g., [data parallelization](https://docs.pytorch.org/docs/stable/elastic/run.html), [multi-node training](ray.io)) in the future.
{{% /hint %}}


## Model Parallelization

When `parallel_training=mp`, CoMLRL requires explicit `agent_devices` / `critic_devices` configuration and deploys the agents and critics accordingly.
The training and inference for each model (agent/critic) are running separately on its assigned device.
The responses are aggregated on the CPU and pass to the reward function. The reward is then broadcast back to all devices for training.
MP supports training larger and more models than a single GPU can hold, but the training throughput is limited by the slowest model.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_iac.py
  --config configs/iac_xxx.yaml
  --override
    agent_model="model_a"
    agents=None
    critic_model="model_b"
    critics=None
    iac.use_separate_critic=true
    iac.parallel_training=mp
    iac.agent_devices='["cuda:0","cuda:1"]'
    iac.critic_devices='["cuda:2","cuda:3"]'
```

{{% hint note %}}
Note that when `parallel_training=mp`, even if the same models with same sampling are used on the same seed, the training is not deterministic due to the non-deterministic GPU scheduling and aggregation on CPU.
{{% /hint %}}
