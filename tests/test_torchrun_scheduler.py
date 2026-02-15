import os
from contextlib import contextmanager

import pytest

from comlrl.schedulers.torchrun_scheduler import TorchrunScheduler


@contextmanager
def _set_env(**updates):
    old = {k: os.environ.get(k) for k in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_auto_uses_mp_when_world_size_is_one():
    with _set_env(
        WORLD_SIZE="1",
        RANK=None,
        LOCAL_RANK=None,
        MASTER_ADDR=None,
        MASTER_PORT=None,
    ):
        assert TorchrunScheduler.resolve_mode("auto") == "mp"


def test_auto_uses_mp_when_torchrun_env_is_incomplete():
    with _set_env(
        WORLD_SIZE="2",
        RANK=None,
        LOCAL_RANK=None,
        MASTER_ADDR=None,
        MASTER_PORT=None,
    ):
        assert TorchrunScheduler.resolve_mode("auto") == "mp"


def test_auto_uses_ddp_when_torchrun_env_is_complete():
    with _set_env(
        WORLD_SIZE="2",
        RANK="0",
        LOCAL_RANK="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
    ):
        assert TorchrunScheduler.resolve_mode("auto") == "ddp"


def test_ddp_requires_complete_torchrun_env():
    with _set_env(
        WORLD_SIZE="2",
        RANK=None,
        LOCAL_RANK=None,
        MASTER_ADDR=None,
        MASTER_PORT=None,
    ):
        with pytest.raises(ValueError, match="Missing"):
            TorchrunScheduler.resolve_mode("ddp")


def test_mp_rejects_world_size_greater_than_one():
    with _set_env(WORLD_SIZE="2"):
        with pytest.raises(ValueError, match="WORLD_SIZE=1"):
            TorchrunScheduler.resolve_mode("mp")
