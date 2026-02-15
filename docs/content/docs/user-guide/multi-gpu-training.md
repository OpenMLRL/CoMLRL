---
title: Multi-GPU Training
weight: 6
---

When multiple GPUs are available, CoMLRL can improve training throughput and reduce training time.

CoMLRL supports two ways to leverage multiple GPUs: Model Parallel scheduling (**MP**) for agent/critic placement in a single process and PyTorch Distributed Data Parallel (**DDP**) across multiple processes

## Model Parallel Scheduler

Single-process model placement that assigns different agents/critics to different GPUs.

- Best for multi-agent layouts or heterogeneous model placement
- No gradient all-reduce across processes
- Configure with `agent_devices` / `critic_devices`

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --config configs/iac_xxx.yaml \
  --override \
    iac.parallel_training=mp \
    iac.agent_devices='["cuda:0","cuda:1"]' \
    iac.critic_devices='["cuda:2","cuda:3"]'
```

## Distributed Data Parallel

Distributed Data Parallel uses multiple processes (typically one process per GPU) and synchronizes gradients.

- Best for scaling data-parallel throughput
- Requires `torchrun` (multi-process launch)
- Does not reduce per-GPU model memory by itself

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --config configs/iac_xxx.yaml \
  --override iac.parallel_training=ddp
```

## `auto` Selection Rule

`parallel_training=auto` is resolved by `WORLD_SIZE`:

- `WORLD_SIZE > 1` -> `ddp`
- `WORLD_SIZE = 1` -> `mp`

So plain `python` (single process) selects `mp`, even when multiple GPUs are visible.

## `CUDA_VISIBLE_DEVICES` vs `WORLD_SIZE`

- `CUDA_VISIBLE_DEVICES`: which GPUs each process can see
- `WORLD_SIZE`: how many processes participate in distributed training

These are related but not the same variable.

## Config Fields

Use these fields in `iac` / `maac` / `magrpo`:

- `parallel_training`: `auto | ddp | mp`
- `agent_devices`: optional device spec (string or list)
- `critic_devices`: optional device spec for IAC/MAAC

{{% hint info %}}
`parallel_training=ddp` and explicit `agent_devices` / `critic_devices` are mutually exclusive.
{{% /hint %}}
