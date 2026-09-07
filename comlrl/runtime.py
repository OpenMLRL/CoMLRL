"""Opt-in launch helpers; importing this module does not initialize CUDA."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Optional


_SHARED_FILESYSTEMS = {
    "nfs",
    "nfs4",
    "lustre",
    "gpfs",
    "beegfs",
    "ceph",
    "cifs",
    "smb3",
    "fuse.sshfs",
    "fuse.ceph",
    "fuse.glusterfs",
}


def _filesystem_type(path: Path) -> Optional[str]:
    """Find the most specific Linux mount, including escaped mount paths."""
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.exists():
        return None
    resolved = str(path.resolve())
    best = (0, None)
    for line in mountinfo.read_text().splitlines():
        fields = line.split()
        if len(fields) < 7 or "-" not in fields:
            continue
        mount = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
        if resolved == mount or resolved.startswith(mount.rstrip("/") + "/"):
            # Overmounts (e.g. autofs followed by NFS at /home) can have the
            # same path. Prefer the later, visible mount in that case.
            if len(mount) >= best[0]:
                best = (len(mount), fields[fields.index("-") + 1])
    return best[1]


def configure_job_cuda_cache() -> Optional[str]:
    """Isolate an unset CUDA JIT cache for a Slurm launch, before CUDA starts.

    Non-Slurm processes and explicit CUDA cache settings are unchanged. A private
    directory is created on node-local storage; subprocesses inherit its path.
    No shared cache is removed, and no model/training configuration is changed.
    """
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not job_id or os.environ.get("CUDA_CACHE_DISABLE") == "1":
        return None
    existing = os.environ.get("CUDA_CACHE_PATH")
    if existing:
        return existing
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError(
            "SLURM_JOB_ID must not contain path separators or special characters."
        )

    # Inspect an already-imported torch only; this helper itself needs no torch.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError(
            "Configure the job CUDA cache before initializing CUDA; restart the launch."
        )

    override = os.environ.get("COMLRL_CUDA_CACHE_ROOT")
    root = Path(override or os.environ.get("SLURM_TMPDIR") or "/tmp")
    if not root.is_absolute():
        raise ValueError(
            "The CUDA cache root must be an absolute node-local directory."
        )
    shared = _filesystem_type(root) in _SHARED_FILESYSTEMS
    if not override and (not root.is_dir() or shared):
        root = Path("/tmp")
        shared = _filesystem_type(root) in _SHARED_FILESYSTEMS
    if shared or not root.is_dir():
        raise RuntimeError(
            "CUDA cache root must be an existing node-local directory, not a shared filesystem. "
            "Set COMLRL_CUDA_CACHE_ROOT to a suitable directory."
        )

    # mkdtemp atomically creates a mode-0700 directory, avoiding predictable
    # /tmp paths and symlink races. A subsequent call reuses the exported path.
    try:
        cache = tempfile.mkdtemp(prefix=f"comlrl-cuda-{job_id}-", dir=str(root))
    except OSError as exc:
        raise RuntimeError(f"Cannot create a private CUDA cache under {root}") from exc
    os.environ["CUDA_CACHE_PATH"] = cache
    print(
        f"[runtime] job={job_id} CUDA_CACHE_PATH={cache}", file=sys.stderr, flush=True
    )
    return cache
