---
title: Parallel Training
weight: 6
---

CoMLRL supports two mutually-exclusive execution schedulers for training/inference:

- `ddp`: Distributed Data Parallel via `torchrun` (`TorchrunScheduler`)
- `scheduler`: Single-process device placement for agents/critics (`DeviceScheduler`)

## Config Fields

Use these fields in `iac` / `maac` / `magrpo` sections:

- `parallel_training`: `auto | ddp | scheduler`
- `agent_devices`: optional device spec (e.g., `"cuda:0"` or `["cuda:0", "cuda:1"]`)
- `critic_devices`: optional device spec for critic(s) (IAC/MAAC)

{{% hint info %}}
`parallel_training=ddp` and explicit `agent_devices` / `critic_devices` are mutually exclusive.
{{% /hint %}}

## How `auto` Is Resolved

`auto` is resolved by `WORLD_SIZE`:

- `WORLD_SIZE > 1` -> `ddp`
- `WORLD_SIZE = 1` -> `scheduler`

This decision does **not** depend on how many GPUs are visible.

## `CUDA_VISIBLE_DEVICES` vs `WORLD_SIZE`

- `CUDA_VISIBLE_DEVICES`: which GPUs are visible to each process
- `WORLD_SIZE`: how many processes participate in distributed training

Examples:

1. `CUDA_VISIBLE_DEVICES=0,1 python train.py ...`
- one process (`WORLD_SIZE=1`)
- `auto` -> `scheduler`

2. `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py ...`
- two processes (`WORLD_SIZE=2`)
- `auto` -> `ddp`

## Usage Examples

### Device Scheduler (single process, model-level placement)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --config configs/iac_xxx.yaml \
  --override \
    iac.parallel_training=scheduler \
    iac.agent_devices='["cuda:0","cuda:1"]' \
    iac.critic_devices='["cuda:2","cuda:3"]'
```

### DDP (multi-process data parallel)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --config configs/iac_xxx.yaml \
  --override iac.parallel_training=ddp
```

For DDP, do not set `agent_devices`/`critic_devices`.

## Logging Note

In DDP mode, trainer metrics and model save are rank-0 only. W&B `system/*` metrics reflect the rank-0 process view by default.
