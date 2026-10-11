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


def configure_cuda_cache(
    root: Optional[str] = None, *, run_id: Optional[str] = None
) -> Optional[str]:
    """Opt in to a private local CUDA JIT cache before initializing CUDA.

    Explicit CUDA cache settings take precedence. The root defaults to
    COMLRL_CUDA_CACHE_ROOT or /tmp; subprocesses inherit the created cache.
    """
    if os.environ.get("CUDA_CACHE_DISABLE") == "1":
        return None
    existing = os.environ.get("CUDA_CACHE_PATH")
    if existing:
        return existing
    if run_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must contain only letters, digits, '_' or '-'.")

    # Inspect an already-imported torch only; this helper itself needs no torch.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError(
            "Configure the CUDA cache before initializing CUDA; restart the launch."
        )

    root = Path(root or os.environ.get("COMLRL_CUDA_CACHE_ROOT") or "/tmp")
    if not root.is_absolute():
        raise ValueError(
            "The CUDA cache root must be an absolute node-local directory."
        )
    shared = _filesystem_type(root) in _SHARED_FILESYSTEMS
    if shared or not root.is_dir():
        raise RuntimeError(
            "CUDA cache root must be an existing node-local directory, not a shared filesystem. "
            "Set COMLRL_CUDA_CACHE_ROOT to a suitable directory."
        )

    # mkdtemp atomically creates a mode-0700 directory, avoiding predictable
    # /tmp paths and symlink races. A subsequent call reuses the exported path.
    try:
        prefix = f"comlrl-cuda-{run_id}-" if run_id is not None else "comlrl-cuda-"
        cache = tempfile.mkdtemp(prefix=prefix, dir=str(root))
    except OSError as exc:
        raise RuntimeError(f"Cannot create a private CUDA cache under {root}") from exc
    os.environ["CUDA_CACHE_PATH"] = cache
    print(f"[runtime] CUDA_CACHE_PATH={cache}", file=sys.stderr, flush=True)
    return cache


def configure_job_cuda_cache() -> Optional[str]:
    """Compatibility wrapper for existing Slurm launchers; otherwise a no-op."""
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not job_id or os.environ.get("CUDA_CACHE_DISABLE") == "1":
        return None
    if os.environ.get("CUDA_CACHE_PATH"):
        return os.environ["CUDA_CACHE_PATH"]

    root = os.environ.get("COMLRL_CUDA_CACHE_ROOT")
    if not root:
        root = os.environ.get("SLURM_TMPDIR")
        if root and Path(root).is_absolute():
            if (
                not Path(root).is_dir()
                or _filesystem_type(Path(root)) in _SHARED_FILESYSTEMS
            ):
                root = None
    return configure_cuda_cache(root, run_id=job_id)
