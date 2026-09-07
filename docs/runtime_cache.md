# Slurm CUDA cache isolation

Call the standard-library-only launch helper before any CUDA initialization:

```python
from comlrl.runtime import configure_job_cuda_cache

configure_job_cuda_cache()
```

The domain training entrypoints call this at the start of `main()`. Update the
CoMLRL checkout/environment together with the domain checkout. Importing CoMLRL
alone does not change the environment or create directories.

- Outside Slurm, nothing changes.
- An explicit `CUDA_CACHE_PATH` or `CUDA_CACHE_DISABLE=1` is respected.
- Otherwise, each launch gets a private mode-0700 cache under `SLURM_TMPDIR` when
  it is an existing local directory, falling back to `/tmp`. Network/shared
  mounts are rejected; `TMPDIR` is deliberately not used as an implicit root.
- `COMLRL_CUDA_CACHE_ROOT` can explicitly select an existing, writable node-local
  directory. An invalid override fails clearly instead of silently using NFS.
- Repeated calls reuse the exported path, and subprocesses inherit it. Different
  launches/jobs do not contend for the same CUDA cache index.
- One `[runtime]` stderr line records the selected path. Reward, seeds, candidate
  counts, training budgets, model loading options and CUDA cache size are unchanged.

The helper never deletes an existing/shared cache. Cache lifetime follows the
cluster's temporary-storage cleanup policy; `/tmp` caches may persist after a
job, so sites without automatic cleanup should provide job-local `SLURM_TMPDIR`.
Do not remove a cache while its processes are still alive.

This fixes future launches from updated sources. Already-running processes and
old frozen source snapshots are unchanged. Updating a shell variable or pulling
a repository cannot redirect an already-initialized CUDA context. New snapshots
must include both the domain hook and this CoMLRL helper.

CPU-only tests:

```bash
python -m unittest discover -s tests -p test_runtime_cache.py -v
```

Background: [NVIDIA CUDA cache environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-cache-path)
and [NVIDIA's NFS cache guidance](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/).
