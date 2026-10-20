"""MADPO, MARLHF and iterative trainers resume generated pairs and phases."""

import random
import shutil

import pytest
import torch
import wandb

from comlrl.trainers.pbrl import (
    MADPOIterTrainer,
    MADPOTrainer,
    MARLHFIterTrainer,
    MARLHFTrainer,
    marlhf,
)
from comlrl.trainers.pbrl.checkpoint_extras import train_iterations_checkpointed

from checkpoint_helpers import (
    Interrupted,
    TinyACTrainer,
    assert_same_training,
    attach_rng_consuming_logger,
    iterative_trainer,
    marlhf_ac_trainer,
    marlhf_trainer,
    pair,
    stop_after,
    toy_trainer,
)
from helpers import assert_nested_equal, rng_state, seed_all


def preflight_trainer(kind, directory):
    if kind == "madpo":
        return toy_trainer()
    if kind == "marlhf":
        return marlhf_trainer()
    return iterative_trainer(
        directory, MARLHFIterTrainer if kind == "marlhf_iter" else MADPOIterTrainer
    )


def saved_progress(kind):
    if kind == "madpo":
        return {
            "kind": kind,
            "epoch": 2,
            "updates_seen": 4,
            "preference_pairs": [pair(1)],
            "complete": False,
        }
    if kind == "marlhf":
        return {
            "kind": kind,
            "phase": "online_rl",
            "rl": {"kind": "magrpo", "epoch": 2},
        }
    return {
        "kind": kind,
        "iteration": 2,
        "updates_seen": 4,
        "total_pairs": 1,
        "complete": False,
    }


def test_madpo_resume_preserves_pairs_shuffle_optimizer_and_rng(tmp_path):
    seed_all(12)
    baseline = toy_trainer()
    baseline.train()
    expected_rng = rng_state()

    seed_all(12)
    interrupted = toy_trainer()
    stop_after(interrupted, lambda progress: progress.get("epoch") == 1)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)

    seed_all(999)
    resumed = toy_trainer()
    resumed._build_preference_dataset = lambda **kwargs: pytest.fail(
        "regenerated pairs"
    )
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoint-2")
    assert_same_training(baseline, resumed)
    assert baseline.events == interrupted.events + resumed.events
    assert_nested_equal(rng_state(), expected_rng)
    assert torch.equal(
        baseline.reference_models[0].weight, resumed.reference_models[0].weight
    )


def test_madpo_commits_generated_pairs_before_first_update_and_large_interval(tmp_path):
    seed_all(12)
    baseline = toy_trainer()
    baseline.train()
    expected_rng = rng_state()
    seed_all(12)
    interrupted = toy_trainer()
    stop_after(interrupted, lambda progress: progress["epoch"] == 0)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path, checkpoint_interval=100)
    assert not interrupted.optimizers[0].state
    assert interrupted.events == []
    assert (tmp_path / "checkpoint-1" / "training_state.pt").is_file()

    seed_all(999)
    resumed = toy_trainer()
    resumed._build_preference_dataset = lambda **kwargs: pytest.fail(
        "regenerated pairs"
    )
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoint-1")
    assert_same_training(baseline, resumed)
    assert baseline.events == resumed.events
    assert_nested_equal(rng_state(), expected_rng)


def test_madpo_first_update_failure_leaves_generated_pairs_resumable(tmp_path):
    trainer = toy_trainer()

    def fail(batch):
        raise Interrupted("first optimizer update failed")

    trainer._update_from_preference_batch = fail
    with pytest.raises(Interrupted):
        trainer.train(checkpoint_dir=tmp_path)
    resumed = toy_trainer()
    progress = resumed.load_checkpoint(tmp_path / "checkpoint-1")
    assert progress["epoch"] == progress["updates_seen"] == 0
    assert len(progress["preference_pairs"]) == 6
    assert not resumed.optimizers[0].state


def test_madpo_empty_generation_has_one_complete_boundary(tmp_path):
    trainer = toy_trainer()
    trainer._build_preference_dataset = lambda **kwargs: []
    trainer.train(checkpoint_dir=tmp_path, checkpoint_interval=100)
    assert trainer._checkpoint_progress["complete"] is True
    assert trainer._checkpoint_progress["preference_pairs"] == []
    assert len(list(tmp_path.glob("checkpoint-*"))) == 1


def test_iterative_resume_moves_history_replay_and_comparator_rng(tmp_path):
    seed_all(25)
    baseline = iterative_trainer(tmp_path / "baseline")
    baseline.train()
    expected_rng = rng_state()

    seed_all(25)
    interrupted = iterative_trainer(tmp_path / "original")
    stop_after(interrupted, lambda progress: progress.get("iteration") == 2)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path / "checkpoints")
    assert (tmp_path / "checkpoints" / "checkpoint-1").is_dir()  # Empty iteration.
    portable = tmp_path / "portable"
    shutil.copytree(tmp_path / "checkpoints" / "checkpoint-2", portable)
    shutil.rmtree(tmp_path / "original")
    shutil.rmtree(tmp_path / "checkpoints")

    seed_all(777)
    resumed = iterative_trainer(tmp_path / "new-run")
    resumed.train(resume_from_checkpoint=portable)
    assert_same_training(baseline, resumed)
    assert baseline.events == interrupted.events + resumed.events
    assert_nested_equal(rng_state(), expected_rng)
    assert [shard.iteration for shard in resumed._preference_replay_shards_state] == [
        2,
        3,
        4,
    ]
    assert torch.equal(
        baseline._comparator_rng_states["cpu"],
        resumed._comparator_rng_states[resumed._comparator_rng_streams["0"]["id"]],
    )
    assert not (tmp_path / "original").exists()


@pytest.mark.parametrize("boundary", ["reward", "online"])
def test_marlhf_resumes_reward_phase_and_online_epoch(tmp_path, boundary):
    seed_all(31)
    baseline = marlhf_trainer()
    baseline.train()
    expected_rng = rng_state()

    seed_all(31)
    interrupted = marlhf_trainer()
    stop_after(
        interrupted,
        lambda progress: progress.get("phase") == "online_rl"
        and (
            not progress.get("rl")
            if boundary == "reward"
            else progress.get("rl", {}).get("epoch") == 1
        ),
    )
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)
    checkpoint = tmp_path / ("checkpoint-1" if boundary == "reward" else "checkpoint-2")

    seed_all(89)
    resumed = marlhf_trainer()
    resumed._build_preference_dataset = lambda **kwargs: pytest.fail(
        "regenerated reward data"
    )
    resumed._train_reward_model = lambda pairs: pytest.fail("retrained reward model")
    resumed.train(resume_from_checkpoint=checkpoint)
    assert_same_training(baseline, resumed)
    assert torch.equal(baseline.reward_model.weight, resumed.reward_model.weight)
    assert baseline.events == interrupted.events + resumed.events
    assert_nested_equal(rng_state(), expected_rng)


def test_marlhf_discarded_reward_optimizer_resume_equals_continuous(tmp_path):
    def make():
        trainer = marlhf_trainer()
        original = trainer._train_reward_model

        def reward_training(pairs):
            original(pairs)
            trainer.reward_optimizer = None

        trainer._train_reward_model = reward_training
        return trainer

    seed_all(47)
    baseline = make()
    baseline.train()
    expected_rng = rng_state()
    seed_all(47)
    interrupted = make()
    stop_after(interrupted, lambda p: p.get("phase") == "online_rl" and not p.get("rl"))
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)
    seed_all(99)
    resumed = make()
    resumed._train_reward_model = lambda pairs: pytest.fail("reward phase repeated")
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoint-1")
    assert resumed.reward_optimizer is None
    assert_same_training(baseline, resumed)
    assert_nested_equal(rng_state(), expected_rng)


@pytest.mark.parametrize("scoring", ["reward_model", "task"])
def test_marlhf_iteration_restores_previous_reward_model_for_next_pairs(
    tmp_path, scoring
):
    seed_all(45)
    baseline = iterative_trainer(tmp_path / "baseline", MARLHFIterTrainer, scoring)
    baseline.train()
    expected_rng = rng_state()

    seed_all(45)
    interrupted = iterative_trainer(
        tmp_path / "interrupted", MARLHFIterTrainer, scoring
    )
    stop_after(interrupted, lambda progress: progress.get("iteration") == 2)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path / "checkpoints")
    seed_all(91)
    resumed = iterative_trainer(tmp_path / "resumed", MARLHFIterTrainer, scoring)
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoints" / "checkpoint-2")
    assert_same_training(baseline, resumed)
    if scoring == "reward_model":
        assert torch.equal(baseline.reward_model.weight, resumed.reward_model.weight)
        for name in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(
                baseline.reward_optimizer.state_dict()["state"][0][name],
                resumed.reward_optimizer.state_dict()["state"][0][name],
            )
    else:
        assert baseline.reward_model is resumed.reward_model is None
        assert baseline.reward_optimizer is resumed.reward_optimizer is None
    assert baseline.events == interrupted.events + resumed.events
    assert_nested_equal(rng_state(), expected_rng)


@pytest.mark.parametrize("boundary", ["reward", "online"])
def test_marlhf_nested_ac_restores_critic_optimizer_references_and_rng(
    tmp_path, monkeypatch, boundary
):
    monkeypatch.setattr(marlhf, "MAACTrainer", TinyACTrainer)
    seed_all(42)
    baseline = marlhf_ac_trainer()
    baseline.train()
    baseline_child = TinyACTrainer.instances[-1]
    expected_rng = rng_state()

    seed_all(42)
    interrupted = marlhf_ac_trainer()
    stop_after(
        interrupted,
        lambda progress: progress.get("phase") == "online_rl"
        and (
            not progress.get("rl")
            if boundary == "reward"
            else progress.get("rl", {}).get("epoch") == 1
        ),
    )
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)
    checkpoint = tmp_path / ("checkpoint-1" if boundary == "reward" else "checkpoint-2")

    seed_all(75)
    resumed = marlhf_ac_trainer()
    resumed.train(resume_from_checkpoint=checkpoint)
    child = resumed._checkpoint_ac_trainer
    assert torch.equal(baseline.agents[0].weight, resumed.agents[0].weight)
    assert torch.equal(baseline_child.critics[0].weight, child.critics[0].weight)
    assert torch.equal(
        baseline_child.reference_models[0].weight, child.reference_models[0].weight
    )
    for name in ("exp_avg", "exp_avg_sq", "step"):
        assert torch.equal(
            baseline_child.critic_optimizer.state_dict()["state"][0][name],
            child.critic_optimizer.state_dict()["state"][0][name],
        )
    assert baseline.env_step == resumed.env_step
    assert_nested_equal(rng_state(), expected_rng)


@pytest.mark.parametrize("kind", ["madpo", "iterative", "marlhf"])
def test_preference_resume_logging_consumes_rng_without_changing_training(
    tmp_path, kind, monkeypatch
):
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)

    def make(label, initialized):
        seed_all(183)
        if kind == "madpo":
            trainer = toy_trainer()
        elif kind == "iterative":
            trainer = iterative_trainer(tmp_path / label)
        else:
            trainer = marlhf_trainer()
        return attach_rng_consuming_logger(trainer, initialized=initialized)

    baseline = make("baseline", True)
    baseline.train()
    expected_rng = rng_state()
    interrupted = make("interrupted", True)
    predicate = (
        (lambda progress: progress.get("iteration") == 2)
        if kind == "iterative"
        else (lambda progress: progress.get("rl", progress).get("epoch") == 1)
    )
    stop_after(interrupted, predicate)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path / "checkpoints")
    resumed = make("resumed", False)
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoints" / "checkpoint-2")
    assert resumed.logging_initializations == 1
    assert_same_training(baseline, resumed)
    assert baseline.events == interrupted.events + resumed.events
    assert_nested_equal(rng_state(), expected_rng)


@pytest.mark.parametrize(
    "cls", [MADPOTrainer, MARLHFTrainer, MADPOIterTrainer, MARLHFIterTrainer]
)
@pytest.mark.parametrize("interval", [0, -1, True, False, 1.0, "1", None])
def test_interval_only_is_validated_before_training(cls, interval):
    trainer = object.__new__(cls)
    with pytest.raises(ValueError, match="checkpoint_interval"):
        trainer.train(checkpoint_interval=interval)


@pytest.mark.parametrize("kind", ["madpo", "marlhf", "madpo_iter", "marlhf_iter"])
def test_plain_preference_training_does_not_retain_checkpoint_data(tmp_path, kind):
    if kind == "madpo":
        trainer = toy_trainer()
    elif kind == "marlhf":
        trainer = marlhf_trainer()
    elif kind == "madpo_iter":
        trainer = iterative_trainer(tmp_path / "run")
    else:
        trainer = toy_trainer(MARLHFIterTrainer)
        trainer.args.num_iterations = 0
    trainer.train()
    assert trainer._checkpoint_untracked
    assert not getattr(trainer, "_checkpoint_progress", {}).get("preference_pairs")
    with pytest.raises(RuntimeError, match="checkpoint_dir"):
        trainer.save_checkpoint(tmp_path / "manual")


def test_extra_state_rejects_arbitrary_attributes():
    trainer = toy_trainer()
    with pytest.raises(ValueError, match="extra-state"):
        trainer._checkpoint_restore_extra_state({"agents": []}, ".")
    assert len(trainer.agents) == 1


def test_missing_replay_asset_rejected_before_model_mutation(tmp_path):
    trainer = iterative_trainer(tmp_path / "run")
    stop_after(trainer, lambda progress: progress.get("iteration") == 2)
    with pytest.raises(Interrupted):
        trainer.train(checkpoint_dir=tmp_path / "checkpoints")
    checkpoint = tmp_path / "checkpoints" / "checkpoint-2"
    next((checkpoint / "assets" / "preference_replay").iterdir()).unlink()
    resumed = iterative_trainer(tmp_path / "new")
    before = resumed.agents[0].weight.detach().clone()
    with pytest.raises(FileNotFoundError):
        resumed.load_checkpoint(checkpoint)
    assert torch.equal(before, resumed.agents[0].weight)


@pytest.mark.parametrize("value", [True, "1", 0.5, -1])
@pytest.mark.parametrize("kind", ["madpo", "marlhf", "madpo_iter"])
def test_preference_checkpoint_cursors_reject_noninteger_or_negative_values(
    tmp_path, kind, value
):
    if kind == "madpo":
        trainer = toy_trainer()
        trainer._checkpoint_progress = {
            "kind": kind,
            "epoch": value,
            "updates_seen": 0,
            "preference_pairs": [pair(1)],
            "complete": False,
        }
        run = trainer._train_madpo_checkpointed
    elif kind == "marlhf":
        trainer = marlhf_trainer()
        trainer._checkpoint_progress = {
            "kind": kind,
            "phase": "online_rl",
            "rl": {"kind": "magrpo", "epoch": value},
        }
        run = trainer._train_marlhf_checkpointed
    else:
        trainer = iterative_trainer(tmp_path)
        trainer._checkpoint_progress = {
            "kind": kind,
            "iteration": value,
            "updates_seen": 0,
            "total_pairs": 0,
            "complete": False,
        }
        run = lambda: train_iterations_checkpointed(trainer, reward_model=False)
    with pytest.raises(ValueError):
        run()


@pytest.mark.parametrize("kind", ["madpo", "marlhf", "madpo_iter"])
def test_preference_checkpoint_cursors_require_complete_shape(tmp_path, kind):
    if kind == "madpo":
        trainer = toy_trainer()
        run = trainer._train_madpo_checkpointed
    elif kind == "marlhf":
        trainer = marlhf_trainer()
        run = trainer._train_marlhf_checkpointed
    else:
        trainer = iterative_trainer(tmp_path)
        run = lambda: train_iterations_checkpointed(trainer, reward_model=False)
    trainer._checkpoint_progress = {"kind": kind}
    with pytest.raises(ValueError):
        run()


@pytest.mark.parametrize("kind", ["madpo", "marlhf", "madpo_iter", "marlhf_iter"])
@pytest.mark.parametrize("damage", ["malformed", "missing", "shortened"])
def test_progress_preflight_rejects_before_restoring_any_state(tmp_path, kind, damage):
    seed_all(101)
    source = preflight_trainer(kind, tmp_path / "source-run")
    source._update_from_preference_batch([pair(4)])
    source.env_step = 99
    source._checkpoint_progress = saved_progress(kind)
    if kind == "marlhf":
        source._init_reward_model()
    cursor = "iteration" if kind.endswith("_iter") else "epoch"
    cursor_progress = source._checkpoint_progress.get("rl", source._checkpoint_progress)
    if damage == "malformed":
        cursor_progress[cursor] = "2"
    elif damage == "missing":
        cursor_progress.pop(cursor)
    checkpoint = tmp_path / "checkpoint"
    source.save_checkpoint(checkpoint)

    seed_all(202)
    resumed = preflight_trainer(kind, tmp_path / "new-run")
    if damage == "shortened":
        if kind.endswith("_iter"):
            resumed.args.num_iterations = 1
        else:
            resumed.args.num_train_epochs = 1
    if kind == "marlhf":
        resumed._init_reward_model = lambda: pytest.fail("initialized before preflight")
    before_weights = resumed.agents[0].weight.detach().clone()
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    with pytest.raises(ValueError):
        resumed.load_checkpoint(checkpoint)
    assert torch.equal(before_weights, resumed.agents[0].weight)
    assert not resumed.optimizers[0].state
    assert resumed.env_step == 0
    assert not hasattr(resumed, "_checkpoint_progress")
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()
    if kind == "marlhf":
        assert resumed.reward_model is None


def test_missing_preference_pairs_rejected_before_weight_restore(tmp_path):
    source = toy_trainer()
    source._checkpoint_progress = saved_progress("madpo")
    source._checkpoint_progress.pop("preference_pairs")
    source.save_checkpoint(tmp_path / "checkpoint")
    resumed = toy_trainer()
    before = resumed.agents[0].weight.detach().clone()
    with pytest.raises(ValueError, match="Incomplete"):
        resumed.load_checkpoint(tmp_path / "checkpoint")
    assert torch.equal(before, resumed.agents[0].weight)


@pytest.mark.parametrize("phase", ["online_rl", "complete"])
def test_marlhf_saved_rl_cursor_requires_reward_model_before_restore(tmp_path, phase):
    source = marlhf_trainer()
    source._checkpoint_progress = saved_progress("marlhf")
    source._checkpoint_progress["phase"] = phase
    source.save_checkpoint(tmp_path / "checkpoint")
    resumed = marlhf_trainer()
    before = resumed.agents[0].weight.detach().clone()
    with pytest.raises(ValueError, match="reward model"):
        resumed.load_checkpoint(tmp_path / "checkpoint")
    assert torch.equal(before, resumed.agents[0].weight)


def test_marlhf_saved_ac_cursor_requires_nested_snapshot_before_restore(tmp_path):
    source = marlhf_ac_trainer()
    source._init_reward_model()
    source._checkpoint_progress = {
        "kind": "marlhf",
        "phase": "online_rl",
        "rl": {"kind": "actor_critic", "epoch": 1},
    }
    source.save_checkpoint(tmp_path / "checkpoint")
    resumed = marlhf_ac_trainer()
    before = resumed.agents[0].weight.detach().clone()
    with pytest.raises(ValueError, match="actor-critic progress"):
        resumed.load_checkpoint(tmp_path / "checkpoint")
    assert torch.equal(before, resumed.agents[0].weight)
    assert resumed.reward_model is None
