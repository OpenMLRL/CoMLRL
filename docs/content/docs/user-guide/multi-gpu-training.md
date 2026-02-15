---
title: Training Parallelization
linkTitle: Training Parallelization
weight: 6
---

When multiple GPUs are available, CoMLRL can improve training throughput and reduce training time.

CoMLRL supports two schedulers for leveraging multiple GPUs: Model Parallelization (**MP**) for agent/critic deployment and PyTorch Distributed Data Parallelization (**DDP**) across multiple processes.

## Concepts

- `CUDA_VISIBLE_DEVICES`: The GPUs visible to the current process.
- `WORLD_SIZE`: Total number of distributed processes participating in one training job.
- `RANK`: Global process index in `[0, WORLD_SIZE-1]`.
- `LOCAL_RANK`: Process index on the current node; used to select the node-local GPU.
- `MASTER_ADDR`: Address of the process-group rendezvous host (usually rank 0 node).
- `MASTER_PORT`: Port on `MASTER_ADDR` used to initialize distributed communication.

## Model Parallelization

When `parallel_training=mp`, CoMLRL deploys the agents and critics across the specified devices via `agent_devices` / `critic_devices`.
The training and inference for each model (agent/critic) are running separately on its assigned device.
The responses are aggregated on the CPU and pass to the reward function. The reward is then broadcast back to all devices for training.
MP supports training larger and more models than a single GPU can hold, but the training throughput is limited by the slowest model.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
  --config configs/iac_xxx.yaml
  --override
    iac.parallel_training=mp
    iac.agent_devices='["cuda:0","cuda:1"]'
    iac.critic_devices='["cuda:2","cuda:3"]'
```

## Distributed Data Parallelization

When `parallel_training=ddp`, CoMLRL launches multiple processes (one per GPU) and synchronizes gradients across them. Each process runs the full training loop across multiple models, but only on its assigned GPU. The model parameters are kept in sync across processes using PyTorch's DDP.
DDP improves the training throughput, but requires more GPU memory since each process holds a full copy of the models. DDP also requires more careful setup (e.g., environment variables, process launching) and may not be compatible with all reward functions.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
  --config configs/iac_xxx.yaml
  --override iac.parallel_training=ddp
```

## Auto Parallelization

The `parallel_training` field is set to `auto` by default.
When users have `WORLD_SIZE=1` and `CUDA_VISIBLE_DEVICES=0`, CoMLRL trainers fall back to single-gpu training on `cuda:0` without launching multiple processes.
When users have multiple GPUs available, and `WORLD_SIZE=1`, CoMLRL trainers use MP to deploy models across the visible GPUs.
When users have multiple GPUs and complete torchrun distributed env vars (`WORLD_SIZE/RANK/LOCAL_RANK/MASTER_ADDR/MASTER_PORT`), CoMLRL trainers use DDP to synchronize training across processes.
These two modes are mutually exclusive.
