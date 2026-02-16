from types import SimpleNamespace

import torch

from comlrl.trainers.actor_critic.ac_base import ActorCriticTrainerBase
from comlrl.utils.distributed import (
    DistributedContext,
    all_gather_objects,
    barrier,
    local_context,
    reduce_metrics_dict,
)


def _ctx() -> DistributedContext:
    return DistributedContext(
        enabled=False,
        is_main=True,
        device=torch.device("cpu"),
    )


def test_local_context_defaults_to_single_process():
    ctx = local_context(torch.device("cpu"))
    assert ctx.enabled is False
    assert ctx.is_main is True


def test_reduce_metrics_dict_passthrough():
    metrics = {"loss": 1.5, "reward": 2.5}
    reduced = reduce_metrics_dict(metrics, _ctx())
    assert reduced == metrics
    assert reduced is not metrics


def test_barrier_and_all_gather_are_noop_in_single_process():
    ctx = _ctx()
    barrier(ctx)
    assert all_gather_objects({"a": 1}, ctx) == [{"a": 1}]


def test_ac_base_log_metrics_logs_directly(monkeypatch):
    trainer = ActorCriticTrainerBase()
    trainer.dist_env = _ctx()
    trainer.wandb_initialized = True
    trainer.env_step = 5
    trainer.args = SimpleNamespace(logging_steps=1)
    trainer._last_train_log_step = -1

    called = {"log": []}

    def _fake_log(metrics, step):  # noqa: ARG001
        called["log"].append(dict(metrics))

    monkeypatch.setattr("wandb.log", _fake_log)

    trainer._log_metrics({"loss": 1.0})
    trainer._log_metrics({"reward": 2.0})
    assert called["log"] == [{"loss": 1.0}, {"reward": 2.0}]


def test_evaluate_calls_collect_rollouts_with_eval_flag():
    class _EvalDummyTrainer(ActorCriticTrainerBase):
        def __init__(self):
            self.eval_dataset = [{"id": 0}, {"id": 1}]
            self.args = SimpleNamespace(eval_batch_size=1, eval_num_samples=1)
            self.wandb_initialized = False
            self.env_step = 0
            self.dist_env = _ctx()
            self.verbose = False
            self._collect_calls = 0
            self.eval_flags = []

        def _collect_rollouts(self, item):  # noqa: ARG002
            self._collect_calls += 1
            self.eval_flags.append(bool(getattr(self, "_in_eval", False)))
            return [
                SimpleNamespace(
                    metadata={},
                    reward=torch.tensor([1.0]),
                    returns=torch.tensor([1.5]),
                    old_value=torch.tensor([0.5]),
                )
            ]

    trainer = _EvalDummyTrainer()
    metrics = trainer.evaluate()
    assert trainer._collect_calls == 1
    assert trainer.eval_flags == [True]
    assert trainer._in_eval is False
    assert "eval/turn_1/reward_mean" in metrics
