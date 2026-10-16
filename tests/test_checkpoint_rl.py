"""MAGRPO and actor-critic training resume at complete-epoch boundaries."""

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from comlrl.trainers.actor_critic.ac_base import ActorCriticTrainerBase
from comlrl.trainers.actor_critic.iac import IACConfig
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig, MAGRPOTrainer
from comlrl.utils.checkpoint import _decode, _encode, capture_trainer_state

from checkpoint_helpers import Interrupted, attach_rng_consuming_logger
from helpers import assert_nested_equal, rng_state, seed_all


class _TinyLoop:
    def _init_fixture(self, stop_epoch=None):
        self.train_dataset = [{"prompt": "one"}, {"prompt": "two"}]
        self.eval_dataset = [{"prompt": "evaluation"}]
        self.agents = [torch.nn.Linear(1, 1)]
        self.agent_devices = [torch.device("cpu")]
        self.reference_models = [torch.nn.Linear(1, 1)]
        self.reference_models[0].requires_grad_(False)
        self.rollout_buffers = [[]]
        self.env_step = 0
        self._last_train_log_step = -1
        self.wandb_config = None
        self.wandb_initialized = False
        self.verbose = False
        self.num_agents = 1
        self.stop_epoch = stop_epoch
        self.events = []

    def get_train_dataloader(self):
        # Iteration consumes PyTorch RNG even with deterministic ordering.
        return DataLoader(self.train_dataset, batch_size=1, collate_fn=lambda x: x)

    def evaluate(self, **kwargs):
        value = random.random() + float(np.random.random()) + torch.rand(()).item()
        self.events.append(("eval", self.env_step, value))
        return {"score": value}

    def _collect(self, item):
        value = random.random() + float(np.random.random()) + torch.rand(()).item()
        self.rollout_buffers[0].append(value)
        self.env_step += 1
        self.events.append(("sample", item["prompt"], self.env_step, value))

    def _update(self, models, optimizers):
        if not self.rollout_buffers[0]:
            return
        target = torch.tensor(self.rollout_buffers[0]).mean()
        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad()
            loss = (model(torch.ones(1, 1)).squeeze() - target).square()
            loss.backward()
            optimizer.step()
        for scheduler in self.lr_schedulers:
            scheduler.step()
        self.rollout_buffers[0].clear()
        should_log = (
            self._should_log_train(self.env_step)
            if isinstance(self, MAGRPOTrainer)
            else self._should_log_train()
        )
        self.events.append(("flush-and-log", self.env_step, should_log))

    def _checkpoint_commit(self, progress, force=False):
        assert self.rollout_buffers == [[]]
        assert self.events[-1][0] in {"flush-and-log", "epoch-end"}
        super()._checkpoint_commit(progress, force=force)
        if progress["epoch"] == self.stop_epoch:
            raise Interrupted()


class _TinyMAGRPO(_TinyLoop, MAGRPOTrainer):
    def __init__(self, stop_epoch=None):
        self.args = MAGRPOConfig(
            num_agents=1,
            num_train_epochs=3,
            num_turns=1,
            eval_interval=1,
            logging_steps=3,
            agent_devices="cpu",
        )
        self._init_fixture(stop_epoch)
        self.optimizers = [torch.optim.AdamW(self.agents[0].parameters(), lr=0.03)]
        self.lr_schedulers = [torch.optim.lr_scheduler.StepLR(self.optimizers[0], 1)]

    def _train_step_returns(self, item, **kwargs):
        assert kwargs == {"custom_option": "preserved"}
        self._collect(item)

    def _drain_ready_buffers(self, ready_agents):
        assert ready_agents == [0]
        self._update(self.agents, self.optimizers)


class _TinyActorCritic(_TinyLoop, ActorCriticTrainerBase):
    def __init__(self, stop_epoch=None):
        self.args = IACConfig(
            num_agents=1,
            num_train_epochs=3,
            num_turns=1,
            eval_interval=1,
            logging_steps=3,
            use_separate_critic=True,
            agent_devices="cpu",
        )
        self._init_fixture(stop_epoch)
        self.critics = [torch.nn.Linear(1, 1)]
        self.agent_optimizers = [
            torch.optim.AdamW(self.agents[0].parameters(), lr=0.03)
        ]
        self.critic_optimizers = [
            torch.optim.AdamW(self.critics[0].parameters(), lr=0.02)
        ]
        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(optimizer, 1)
            for optimizer in self.agent_optimizers + self.critic_optimizers
        ]

    def _run_batch(self, batch, epoch_metrics):
        for item in batch:
            self._collect(item)

    def _flush_buffers(self, epoch_metrics):
        self._update(
            self.agents + self.critics,
            self.agent_optimizers + self.critic_optimizers,
        )

    def _on_epoch_end(self, epoch, total_epochs, epoch_metrics):
        self.events.append(("epoch-end", epoch))


def _run(trainer, **kwargs):
    if isinstance(trainer, _TinyMAGRPO):
        kwargs["custom_option"] = "preserved"
    trainer.train(**kwargs)


def _make(trainer_class, stop_epoch=None):
    seed_all(32)
    trainer = trainer_class(stop_epoch)
    seed_all(73)
    return trainer


class _EpochAware:
    def __init__(self, stop_epoch=None):
        super().__init__(stop_epoch)
        self.start_epochs = []
        self.dataloader_epochs = []
        self.next_epoch = 0

    def _on_checkpoint_train_start(self, start_epoch, total_epochs):
        super()._on_checkpoint_train_start(start_epoch, total_epochs)
        self.start_epochs.append((start_epoch, total_epochs))
        self.next_epoch = start_epoch

    def get_train_dataloader(self):
        self.dataloader_epochs.append(self.next_epoch)
        self.next_epoch += 1
        return super().get_train_dataloader()


class _EpochAwareMAGRPO(_EpochAware, _TinyMAGRPO):
    pass


class _EpochAwareActorCritic(_EpochAware, _TinyActorCritic):
    pass


EPOCH_AWARE_TRAINERS = [_EpochAwareMAGRPO, _EpochAwareActorCritic]


@pytest.mark.parametrize("trainer_class", [_TinyMAGRPO, _TinyActorCritic])
def test_epoch_resume_matches_uninterrupted_training(trainer_class, tmp_path):
    uninterrupted = _make(trainer_class)
    _run(uninterrupted)
    expected_rng = rng_state()

    interrupted = _make(trainer_class, stop_epoch=1)
    with pytest.raises(Interrupted):
        _run(interrupted, checkpoint_dir=tmp_path / "initial")
    checkpoint = tmp_path / "initial" / "checkpoint-1"
    assert (checkpoint / "training_state.pt").is_file()

    resumed = _make(trainer_class)
    seed_all(900)  # All three streams must be restored, not reseeded.
    _run(
        resumed,
        resume_from_checkpoint=checkpoint,
        checkpoint_dir=tmp_path / "resumed",
    )

    assert interrupted.events + resumed.events == uninterrupted.events
    assert resumed.env_step == uninterrupted.env_step
    assert resumed._last_train_log_step == uninterrupted._last_train_log_step
    assert resumed.rollout_buffers == [[]]
    assert resumed._checkpoint_progress == uninterrupted._checkpoint_progress
    for name in (
        "agents",
        "critics",
        "reference_models",
        "optimizers",
        "agent_optimizers",
        "critic_optimizers",
        "lr_schedulers",
    ):
        for actual, expected in zip(
            getattr(resumed, name, []), getattr(uninterrupted, name, [])
        ):
            assert_nested_equal(actual.state_dict(), expected.state_dict())
    assert_nested_equal(rng_state(), expected_rng)


@pytest.mark.parametrize("kind", ["magrpo", "actor_critic"])
def test_rl_resume_logging_consumes_rng_without_changing_training(tmp_path, kind):
    trainer_class = _TinyMAGRPO if kind == "magrpo" else _TinyActorCritic
    baseline = attach_rng_consuming_logger(_make(trainer_class), initialized=True)
    _run(baseline)
    expected_rng = rng_state()

    interrupted = attach_rng_consuming_logger(_make(trainer_class, 1), initialized=True)
    with pytest.raises(Interrupted):
        _run(interrupted, checkpoint_dir=tmp_path)
    resumed = attach_rng_consuming_logger(_make(trainer_class), initialized=False)
    _run(resumed, resume_from_checkpoint=tmp_path / "checkpoint-1")

    assert resumed.logging_initializations == 1
    assert baseline.events == interrupted.events + resumed.events
    for name in (
        "agents",
        "critics",
        "optimizers",
        "agent_optimizers",
        "critic_optimizers",
    ):
        for left, right in zip(getattr(baseline, name, []), getattr(resumed, name, [])):
            assert_nested_equal(left.state_dict(), right.state_dict())
    assert_nested_equal(rng_state(), expected_rng)


@pytest.mark.parametrize("trainer_class", EPOCH_AWARE_TRAINERS)
def test_resume_aligns_subclass_before_first_dataloader(trainer_class, tmp_path):
    baseline = _make(trainer_class)
    _run(baseline)
    expected_rng = rng_state()
    assert baseline.start_epochs == []  # The default train path stays unchanged.
    assert baseline.dataloader_epochs == [0, 1, 2]

    interrupted = _make(trainer_class, stop_epoch=1)
    with pytest.raises(Interrupted):
        _run(interrupted, checkpoint_dir=tmp_path)
    assert interrupted.start_epochs == [(0, 3)]
    assert interrupted.dataloader_epochs == [0]

    resumed = _make(trainer_class)
    seed_all(900)
    _run(resumed, resume_from_checkpoint=tmp_path / "checkpoint-1")
    assert resumed.start_epochs == [(1, 3)]
    assert resumed.dataloader_epochs == [1, 2]
    assert resumed.next_epoch == 3
    assert interrupted.events + resumed.events == baseline.events
    for name in (
        "agents",
        "critics",
        "reference_models",
        "optimizers",
        "agent_optimizers",
        "critic_optimizers",
        "lr_schedulers",
    ):
        for actual, expected in zip(
            getattr(resumed, name, []), getattr(baseline, name, [])
        ):
            assert_nested_equal(actual.state_dict(), expected.state_dict())
    assert_nested_equal(rng_state(), expected_rng)


@pytest.mark.parametrize("trainer_class", EPOCH_AWARE_TRAINERS)
def test_completed_checkpoint_still_aligns_counter_without_loading_data(
    trainer_class, tmp_path
):
    completed = _make(trainer_class)
    _run(completed)
    completed.save_checkpoint(tmp_path / "complete")

    resumed = _make(trainer_class)
    _run(resumed, resume_from_checkpoint=tmp_path / "complete")
    assert resumed.start_epochs == [(3, 3)]
    assert resumed.next_epoch == 3
    assert resumed.dataloader_epochs == []
    assert resumed.events == []


@pytest.mark.parametrize("trainer_class", [_TinyMAGRPO, _TinyActorCritic])
def test_checkpoint_interval_still_saves_final_epoch(trainer_class, tmp_path):
    trainer = _make(trainer_class)
    _run(trainer, checkpoint_dir=tmp_path, checkpoint_interval=2)
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-2",
        "checkpoint-3",
    ]


@pytest.mark.parametrize("trainer_class", [_TinyMAGRPO, _TinyActorCritic])
def test_manual_save_after_default_train_preserves_completed_cursor(
    trainer_class, tmp_path
):
    trainer = _make(trainer_class)
    _run(trainer)
    expected_rng = rng_state()
    trainer.save_checkpoint(tmp_path / "manual")
    assert_nested_equal(rng_state(), expected_rng)

    loaded = _make(trainer_class)
    _run(loaded, resume_from_checkpoint=tmp_path / "manual")
    assert loaded.events == []
    assert loaded._checkpoint_progress["epoch"] == 3
    assert_nested_equal(rng_state(), expected_rng)


@pytest.mark.parametrize("trainer_class", [_TinyMAGRPO, _TinyActorCritic])
def test_initial_checkpoint_with_empty_cursor_remains_loadable(trainer_class, tmp_path):
    source = _make(trainer_class)
    source.save_checkpoint(tmp_path / "initial")
    target = _make(trainer_class)

    target.load_checkpoint(tmp_path / "initial")

    assert target._checkpoint_progress == {}
    assert target._checkpoint_validate_progress(capture_trainer_state(target)) == 0
    assert target.env_step == 0


@pytest.mark.parametrize("trainer_class", [_TinyMAGRPO, _TinyActorCritic])
def test_nested_default_train_preserves_parent_progress(trainer_class):
    trainer = _make(trainer_class)
    trainer._checkpoint_context_active = True
    trainer._checkpoint_progress = {"kind": "preference", "iteration": 2}
    _run(trainer)
    assert trainer._checkpoint_progress == {"kind": "preference", "iteration": 2}


@pytest.mark.parametrize("trainer_class", [_TinyMAGRPO, _TinyActorCritic])
@pytest.mark.parametrize("checkpoint_enabled", [False, True])
def test_failed_epoch_cannot_be_saved_with_stale_progress(
    trainer_class, checkpoint_enabled, tmp_path
):
    trainer = _make(trainer_class)
    original_update = trainer._update

    def interrupt_after_flush(*args):
        original_update(*args)
        if trainer.env_step == 4:
            raise Interrupted()

    trainer._update = interrupt_after_flush
    options = {"checkpoint_dir": tmp_path / "run"} if checkpoint_enabled else {}
    with pytest.raises(Interrupted):
        _run(trainer, **options)
    assert trainer.rollout_buffers == [[]]
    with pytest.raises(RuntimeError):
        trainer.save_checkpoint(tmp_path / "unsafe")
    assert not (tmp_path / "unsafe").exists()

    # A later successful fresh run establishes a valid boundary again.
    trainer._update = original_update
    _run(trainer)
    trainer.save_checkpoint(tmp_path / "completed")
    assert trainer._checkpoint_progress["epoch"] == 3
    assert (tmp_path / "completed" / "training_state.pt").is_file()


@pytest.mark.parametrize("trainer_class", [_TinyMAGRPO, _TinyActorCritic])
@pytest.mark.parametrize("epoch", [-1, 4, 1.5, True])
def test_rejects_impossible_epoch_cursor(trainer_class, epoch):
    trainer = _make(trainer_class)
    kind = "magrpo" if trainer_class is _TinyMAGRPO else "actor_critic"
    before = rng_state()
    with pytest.raises(ValueError, match="checkpoint epoch"):
        trainer._train_checkpoint_epochs(
            {"kind": kind, "epoch": epoch}, lambda *args, **kwargs: None
        )
    assert trainer.events == []
    assert_nested_equal(rng_state(), before)


@pytest.mark.parametrize("trainer_class", [_TinyMAGRPO, _TinyActorCritic])
def test_rejects_wrong_loop_and_unflushed_boundary(trainer_class):
    trainer = _make(trainer_class)
    with pytest.raises(ValueError, match="epoch cursor"):
        trainer._train_checkpoint_epochs(
            {"kind": "preference", "epoch": 1}, lambda *args, **kwargs: None
        )
    trainer.rollout_buffers[0].append(0.2)
    kind = "magrpo" if trainer_class is _TinyMAGRPO else "actor_critic"
    with pytest.raises(ValueError, match="empty rollout buffers"):
        trainer._train_checkpoint_epochs(
            {"kind": kind, "epoch": 1}, lambda *args, **kwargs: None
        )


@pytest.mark.parametrize("trainer_class", EPOCH_AWARE_TRAINERS)
def test_invalid_cursor_does_not_invoke_subclass_hook(trainer_class):
    trainer = _make(trainer_class)
    kind = "magrpo" if isinstance(trainer, _TinyMAGRPO) else "actor_critic"
    with pytest.raises(ValueError, match="between 0 and num_train_epochs"):
        trainer._train_checkpoint_epochs(
            {"kind": kind, "epoch": 4}, lambda *args, **kwargs: None
        )
    assert trainer.start_epochs == []
    assert trainer.dataloader_epochs == []


@pytest.mark.parametrize("trainer_class", [_TinyMAGRPO, _TinyActorCritic])
@pytest.mark.parametrize("interval", [True, 1.0, 0, -1, "1"])
def test_invalid_checkpoint_interval_is_rejected_without_other_options(
    trainer_class, interval
):
    trainer = _make(trainer_class)
    before = rng_state()
    with pytest.raises(ValueError, match="checkpoint_interval"):
        _run(trainer, checkpoint_interval=interval)
    assert trainer.events == []
    assert_nested_equal(rng_state(), before)


@pytest.mark.parametrize("trainer_class", [_TinyMAGRPO, _TinyActorCritic])
@pytest.mark.parametrize(
    "incompatibility",
    ["shortened-total", "wrong-kind", "invalid-epoch", "nonempty-buffer"],
)
def test_resume_rejects_incompatible_saved_cursor_before_mutation(
    trainer_class, incompatibility, tmp_path
):
    source = _make(trainer_class)
    _run(source)
    checkpoint = tmp_path / "completed"
    source.save_checkpoint(checkpoint)
    state_path = checkpoint / "training_state.pt"
    state = _decode(torch.load(state_path, weights_only=True, map_location="cpu"))
    if incompatibility == "wrong-kind":
        state["progress"]["kind"] = "preference"
    elif incompatibility == "invalid-epoch":
        state["progress"]["epoch"] = True
    elif incompatibility == "nonempty-buffer":
        state["runtime"]["rollout_buffers"] = [[0.2]]
    if incompatibility != "shortened-total":
        torch.save(_encode(state), state_path)
    target = _make(trainer_class)
    if incompatibility == "shortened-total":
        target.args.num_train_epochs = 2
    seed_all(903)
    before = _encode(capture_trainer_state(target))

    with pytest.raises(ValueError, match="epoch|cursor|empty rollout buffers"):
        _run(target, resume_from_checkpoint=checkpoint)

    assert target.events == []
    assert_nested_equal(_encode(capture_trainer_state(target)), before)
