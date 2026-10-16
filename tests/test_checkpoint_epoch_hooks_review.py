"""Subclass epoch counters follow the validated checkpoint cursor."""

import pytest

from test_checkpoint_rl import (
    _Interrupted,
    _TinyActorCritic,
    _TinyMAGRPO,
    _assert_rng_equal,
    _assert_state_equal,
    _make,
    _rng_state,
    _run,
    _seed,
)


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


TRAINERS = [_EpochAwareMAGRPO, _EpochAwareActorCritic]


@pytest.mark.parametrize("trainer_class", TRAINERS)
def test_resume_aligns_subclass_before_first_dataloader(trainer_class, tmp_path):
    baseline = _make(trainer_class)
    _run(baseline)
    expected_rng = _rng_state()
    assert baseline.start_epochs == []  # The default train path stays unchanged.
    assert baseline.dataloader_epochs == [0, 1, 2]

    interrupted = _make(trainer_class, stop_epoch=1)
    with pytest.raises(_Interrupted):
        _run(interrupted, checkpoint_dir=tmp_path)
    assert interrupted.start_epochs == [(0, 3)]
    assert interrupted.dataloader_epochs == [0]

    resumed = _make(trainer_class)
    _seed(900)
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
            _assert_state_equal(actual.state_dict(), expected.state_dict())
    _assert_rng_equal(_rng_state(), expected_rng)


@pytest.mark.parametrize("trainer_class", TRAINERS)
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


@pytest.mark.parametrize("trainer_class", TRAINERS)
def test_invalid_cursor_does_not_invoke_subclass_hook(trainer_class):
    trainer = _make(trainer_class)
    kind = "magrpo" if isinstance(trainer, _TinyMAGRPO) else "actor_critic"
    with pytest.raises(ValueError, match="between 0 and num_train_epochs"):
        trainer._train_checkpoint_epochs(
            {"kind": kind, "epoch": 4}, lambda *args, **kwargs: None
        )
    assert trainer.start_epochs == []
    assert trainer.dataloader_epochs == []
