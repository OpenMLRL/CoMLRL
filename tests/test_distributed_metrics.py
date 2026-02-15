from types import SimpleNamespace

import torch

import comlrl.utils.distributed as dist_utils
from comlrl.trainers.actor_critic.ac_base import ActorCriticTrainerBase
from comlrl.utils.distributed import DistributedContext


def _ctx(*, enabled: bool, is_main: bool, rank: int = 0, world_size: int = 1):
    return DistributedContext(
        enabled=enabled,
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        is_main=is_main,
        device=torch.device("cpu"),
    )


def test_reduce_metrics_dict_local_returns_input():
    ctx = _ctx(enabled=False, is_main=True)
    metrics = {"loss": 1.5, "reward": 2.5}
    assert dist_utils.reduce_metrics_dict(metrics, ctx) == metrics


def test_reduce_metrics_dict_distributed_main_averages(monkeypatch):
    ctx = _ctx(enabled=True, is_main=True, world_size=2)
    metrics = {"a": 1.0, "b": 3.0}

    monkeypatch.setattr(
        dist_utils,
        "all_gather_objects",
        lambda obj, _ctx: [obj, obj],
    )

    def _fake_all_reduce(tensor, op=None):  # noqa: ARG001
        tensor += torch.tensor([3.0, 1.0], dtype=tensor.dtype, device=tensor.device)

    monkeypatch.setattr(dist_utils.dist, "all_reduce", _fake_all_reduce)

    reduced = dist_utils.reduce_metrics_dict(metrics, ctx)
    assert reduced == {"a": 2.0, "b": 2.0}


def test_reduce_metrics_dict_distributed_non_main_returns_empty(monkeypatch):
    ctx = _ctx(enabled=True, is_main=False, rank=1, world_size=2)
    metrics = {"a": 1.0}

    monkeypatch.setattr(
        dist_utils,
        "all_gather_objects",
        lambda obj, _ctx: [obj, obj],
    )

    def _fake_all_reduce(tensor, op=None):  # noqa: ARG001
        tensor += torch.tensor([1.0], dtype=tensor.dtype, device=tensor.device)

    monkeypatch.setattr(dist_utils.dist, "all_reduce", _fake_all_reduce)

    assert dist_utils.reduce_metrics_dict(metrics, ctx) == {}


def test_ac_base_log_metrics_skips_reduction_when_unsynchronized(monkeypatch):
    trainer = ActorCriticTrainerBase()
    trainer.dist_env = _ctx(enabled=True, is_main=True, world_size=2)
    trainer.wandb_initialized = True
    trainer.env_step = 5
    trainer.args = SimpleNamespace(logging_steps=1)
    trainer._last_train_log_step = -1

    called = {"reduce": 0, "log": []}

    def _fake_reduce(metrics, _ctx):  # noqa: ARG001
        called["reduce"] += 1
        return {"loss": 99.0}

    def _fake_log(metrics, step):  # noqa: ARG001
        called["log"].append(dict(metrics))

    monkeypatch.setattr(
        "comlrl.trainers.actor_critic.ac_base.reduce_metrics_dict", _fake_reduce
    )
    monkeypatch.setattr("wandb.log", _fake_log)

    trainer._log_metrics({"loss": 1.0}, synchronize=False)
    assert called["reduce"] == 0
    assert called["log"] == [{"loss": 1.0}]


def test_ac_base_log_metrics_reduces_when_synchronized(monkeypatch):
    trainer = ActorCriticTrainerBase()
    trainer.dist_env = _ctx(enabled=True, is_main=True, world_size=2)
    trainer.wandb_initialized = True
    trainer.env_step = 6
    trainer.args = SimpleNamespace(logging_steps=1)
    trainer._last_train_log_step = -1

    called = {"reduce": 0, "log": []}

    def _fake_reduce(metrics, _ctx):  # noqa: ARG001
        called["reduce"] += 1
        assert metrics == {"loss": 1.0}
        return {"loss": 2.0}

    def _fake_log(metrics, step):  # noqa: ARG001
        called["log"].append(dict(metrics))

    monkeypatch.setattr(
        "comlrl.trainers.actor_critic.ac_base.reduce_metrics_dict", _fake_reduce
    )
    monkeypatch.setattr("wandb.log", _fake_log)

    trainer._log_metrics({"loss": 1.0}, synchronize=True)
    assert called["reduce"] == 1
    assert called["log"] == [{"loss": 2.0}]
