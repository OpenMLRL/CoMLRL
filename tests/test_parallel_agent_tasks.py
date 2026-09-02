import threading
import time
from types import SimpleNamespace

from comlrl.trainers.actor_critic.ac_base import ActorCriticTrainerBase
from comlrl.trainers.actor_critic.iac import IACTrainer
from comlrl.trainers.actor_critic.maac import MAACTrainer
from comlrl.trainers.reinforce.magrpo import MAGRPOTrainer


class _DummyACTrainer(ActorCriticTrainerBase):
    def __init__(self, *, parallel_training: str, num_agents: int = 2):
        self.parallel_training = parallel_training
        self.args = SimpleNamespace(num_agents=num_agents)
        self.agent_devices = [f"cuda:{idx}" for idx in range(num_agents)]


def test_ac_run_agent_tasks_keeps_index_order_in_mp():
    trainer = _DummyACTrainer(parallel_training="mp", num_agents=2)
    completion_order = []

    def _task(agent_idx: int) -> str:
        if agent_idx == 0:
            time.sleep(0.05)
        completion_order.append(agent_idx)
        return f"agent-{agent_idx}"

    outputs = trainer._run_agent_tasks(_task)
    assert outputs == ["agent-0", "agent-1"]
    assert completion_order == [1, 0]


def test_ac_run_agent_tasks_is_sequential_when_mp_disabled():
    trainer = _DummyACTrainer(parallel_training="none", num_agents=2)
    completion_order = []

    def _task(agent_idx: int) -> int:
        completion_order.append(agent_idx)
        return agent_idx

    outputs = trainer._run_agent_tasks(_task)
    assert outputs == [0, 1]
    assert completion_order == [0, 1]


def test_iac_parallel_updates_enabled_only_for_mp_mode():
    trainer = IACTrainer.__new__(IACTrainer)
    trainer.args = SimpleNamespace(num_agents=2)
    trainer.agent_devices = ["cuda:0", "cuda:1"]
    trainer.parallel_training = "none"
    assert trainer._parallel_agent_mode_enabled() is False
    trainer.parallel_training = "mp"
    assert trainer._parallel_agent_mode_enabled() is True


def test_magrpo_run_agent_tasks_keeps_index_order_in_mp():
    trainer = MAGRPOTrainer.__new__(MAGRPOTrainer)
    trainer.parallel_training = "mp"
    trainer.num_agents = 2
    trainer.agent_devices = ["cuda:0", "cuda:1"]
    completion_order = []

    def _task(agent_idx: int) -> str:
        if agent_idx == 0:
            time.sleep(0.05)
        completion_order.append(agent_idx)
        return f"agent-{agent_idx}"

    try:
        outputs = trainer._run_agent_tasks(_task)
        assert outputs == ["agent-0", "agent-1"]
        assert completion_order == [1, 0]
    finally:
        trainer._shutdown_agent_task_executors()


def test_magrpo_reuses_one_worker_thread_per_agent():
    trainer = MAGRPOTrainer.__new__(MAGRPOTrainer)
    trainer.parallel_training = "mp"
    trainer.num_agents = 2
    trainer.agent_devices = ["cuda:0", "cuda:1"]

    try:
        thread_ids_by_call = [
            trainer._run_agent_tasks(lambda _agent_idx: threading.get_ident())
            for _ in range(8)
        ]

        assert all(
            thread_ids == thread_ids_by_call[0] for thread_ids in thread_ids_by_call[1:]
        )
        assert len(set(thread_ids_by_call[0])) == 2
    finally:
        trainer._shutdown_agent_task_executors()


def test_maac_parallel_updates_are_always_serialized():
    trainer = MAACTrainer.__new__(MAACTrainer)
    trainer._parallel_update_enabled = False
    trainer.parallel_training = "mp"
    trainer.args = SimpleNamespace(num_agents=2)
    trainer.agent_devices = ["cuda:0", "cuda:1"]
    run_parallel = bool(
        getattr(
            trainer,
            "_parallel_update_enabled",
            trainer._parallel_agent_mode_enabled(),
        )
    )
    assert run_parallel is False
