# Optional CUDA cache isolation

CoMLRL runs directly with Python. Cache isolation is optional and independent of
the job scheduler; importing the library does not create directories or change
the environment.

To keep a launch's CUDA JIT cache separate from other processes, call this helper
before initializing CUDA:

```python
from comlrl.runtime import configure_cuda_cache

configure_cuda_cache()
```

`configure_cuda_cache(root=None, *, run_id=None)` uses an explicit `root`, then
`COMLRL_CUDA_CACHE_ROOT`, then `/tmp`. The root must be an existing absolute
directory on local storage. The optional `run_id` labels the directory and may
contain letters, digits, underscores, and hyphens.

- Explicit `CUDA_CACHE_PATH` and `CUDA_CACHE_DISABLE=1` settings are respected.
- A new cache directory has private mode-0700 permissions. Repeated calls reuse
  its exported path, which subprocesses inherit.
- Shared filesystems are rejected. `TMPDIR` is not used implicitly because it
  can point to shared storage.
- The helper reports the selected path once and never deletes existing caches.
  Remove a cache only after all processes using it have stopped.

This changes cache placement, not rewards, seeds, training configuration, or CUDA
cache size. It cannot redirect an already-initialized CUDA context.

## Compatibility with existing launchers

`configure_job_cuda_cache()` remains available for existing Slurm launchers. It
reads `SLURM_JOB_ID` (or `SLURM_JOBID`) and prefers a local `SLURM_TMPDIR`, falling
back to `/tmp`; `COMLRL_CUDA_CACHE_ROOT` takes precedence. Outside Slurm it does
nothing. New Python programs can use the scheduler-independent helper above.

Trainer output paths no longer gain an implicit `job_<id>` subdirectory from the
environment. To retain an existing directory layout, pass the complete run path
as `wandb_config["output_dir"]`, including any desired job subdirectory. The
existing domain preference launchers already pass this path explicitly.

CPU-only checks:

```bash
python -m unittest discover -s tests -p test_runtime_cache.py -v
```
