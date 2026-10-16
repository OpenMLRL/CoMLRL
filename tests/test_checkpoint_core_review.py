"""Regression coverage for checkpoint review items 2, 5, 7, 8 and 10."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import comlrl.utils.checkpoint as checkpoint
from comlrl.utils.wandb_utils import wandb_init_kwargs
from test_checkpoint_state import TinyTrainer, assert_nested_equal
from test_checkpoint_preference import (
    Interrupted,
    assert_same_training,
    marlhf_trainer,
    seed,
    stop_after,
)


@pytest.mark.parametrize("lazy", [False, True])
def test_absent_reward_optimizer_is_authoritative(tmp_path, lazy):
    source = TinyTrainer()
    source.reward_optimizer = None
    state = checkpoint.capture_trainer_state(source)
    target = TinyTrainer()
    if lazy:
        target.reward_model = target.reward_optimizer = None

        def initialize():
            target.reward_model = torch.nn.Linear(3, 1)
            target.reward_optimizer = torch.optim.AdamW(
                target.reward_model.parameters()
            )

        target._init_reward_model = initialize
    checkpoint.restore_trainer_state(target, state, tmp_path)
    assert target.reward_optimizer is None
    assert_nested_equal(
        target.reward_model.state_dict(), source.reward_model.state_dict()
    )


def test_marlhf_discarded_reward_optimizer_resume_equals_continuous(tmp_path):
    def make():
        trainer = marlhf_trainer()
        original = trainer._train_reward_model

        def reward_training(pairs):
            original(pairs)
            trainer.reward_optimizer = None

        trainer._train_reward_model = reward_training
        return trainer

    seed(47)
    baseline = make()
    baseline.train()
    expected_rng = torch.get_rng_state()
    seed(47)
    interrupted = make()
    stop_after(interrupted, lambda p: p.get("phase") == "online_rl" and not p.get("rl"))
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)
    seed(99)
    resumed = make()
    resumed._train_reward_model = lambda pairs: pytest.fail("reward phase repeated")
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoint-1")
    assert resumed.reward_optimizer is None
    assert_same_training(baseline, resumed)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_all_assets_fsynced_before_publish_and_parent_after(tmp_path, monkeypatch):
    trainer = TinyTrainer()
    events = []
    inode_names = {}

    def assets(root):
        asset = root / "assets" / "policy" / "model.safetensors"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"weights")
        for path in [asset, asset.parent, asset.parent.parent, root]:
            inode_names[path.stat().st_ino] = path.name
        return {}

    trainer._checkpoint_save_assets = assets
    original_fsync, original_replace = os.fsync, os.replace

    def fsync(fd):
        info = os.fstat(fd)
        events.append(("sync", inode_names.get(info.st_ino, info.st_ino)))
        original_fsync(fd)

    def replace(source, destination):
        events.append(("rename", Path(destination).name))
        return original_replace(source, destination)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    trainer.save_checkpoint(tmp_path / "checkpoint")
    published = events.index(("rename", "checkpoint"))
    for name in ["model.safetensors", "policy", "assets"]:
        assert ("sync", name) in events[:published]
    assert events[-1] == ("sync", tmp_path.stat().st_ino)


def test_asset_sync_failure_does_not_replace_previous_checkpoint(tmp_path, monkeypatch):
    trainer = TinyTrainer()
    destination = tmp_path / "checkpoint"
    trainer.save_checkpoint(destination)
    previous = (destination / checkpoint.STATE_FILENAME).read_bytes()
    monkeypatch.setattr(
        checkpoint,
        "_fsync_tree",
        lambda root: (_ for _ in ()).throw(OSError("sync failed")),
    )
    trainer.env_step += 1
    with pytest.raises(OSError, match="sync failed"):
        trainer.save_checkpoint(destination)
    assert (destination / checkpoint.STATE_FILENAME).read_bytes() == previous
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_retention_only_prunes_owned_snapshots_after_success(tmp_path):
    unrelated = tmp_path / "checkpoint-900"
    malformed = tmp_path / "checkpoint-901"
    trainer = TinyTrainer()
    with trainer._checkpoint_context(checkpoint_dir=tmp_path, checkpoint_total_limit=2):
        unrelated.mkdir()
        (unrelated / "keep.txt").write_text("not a training checkpoint")
        malformed.mkdir()
        (malformed / checkpoint._OWNER_FILENAME).write_text("[]")
        for epoch in range(4):
            trainer._checkpoint_commit({"epoch": epoch + 1})
    assert sorted(
        path.name
        for path in tmp_path.glob("checkpoint-*")
        if (path / checkpoint.STATE_FILENAME).exists()
    ) == ["checkpoint-3", "checkpoint-4"]
    assert (unrelated / "keep.txt").exists()
    assert malformed.exists()
    # A subsequent resume also preflights future names; keep these unrelated
    # directories outside the automatic snapshot namespace before resuming.
    unrelated.rename(tmp_path / "unrelated")
    malformed.rename(tmp_path / "malformed")
    resumed = TinyTrainer()
    with resumed._checkpoint_context(
        resume_from_checkpoint=tmp_path / "checkpoint-4",
        checkpoint_dir=tmp_path,
        checkpoint_total_limit=1,
    ):
        resumed._checkpoint_commit({"epoch": 5})
    assert (tmp_path / "checkpoint-5" / checkpoint.STATE_FILENAME).exists()
    assert not (tmp_path / "checkpoint-4").exists()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_retention_limit_rejected_before_training(tmp_path, limit):
    trainer = TinyTrainer()
    before = checkpoint.capture_trainer_state(trainer)
    with pytest.raises(ValueError, match="checkpoint_total_limit"):
        with trainer._checkpoint_context(
            checkpoint_dir=tmp_path, checkpoint_total_limit=limit
        ):
            pytest.fail("invalid retention started training")
    assert_nested_equal(checkpoint.capture_trainer_state(trainer), before)


def test_portable_placement_fields_accept_legacy_snapshot(tmp_path):
    source = TinyTrainer()
    source.args.comparator_devices = "cuda:2"
    state = checkpoint.capture_trainer_state(source)
    # Version-1 snapshots written before these fields became portable.
    state["config"].update(comparator_devices="cuda:2", parallel_training="mp")
    target = TinyTrainer()
    target.args.comparator_devices = "cuda:1"
    target.args.parallel_training = "none"
    checkpoint.restore_trainer_state(target, state, tmp_path)
    assert target.args.comparator_devices == "cuda:1"


class FakeRun:
    def __init__(self, id):
        self.id, self.project, self.entity = id, "research", "team"
        self.finished = False

    def finish(self):
        self.finished = True


@pytest.mark.parametrize("eager", [False, True])
def test_wandb_restores_owned_run_identity_and_init_arguments(tmp_path, eager):
    source = TinyTrainer()
    source._checkpoint_record_wandb_run(FakeRun("saved-id"))
    state = checkpoint.capture_trainer_state(source)
    target = TinyTrainer()
    target.wandb_config = {"name": "experiment"}
    target.wandb_initialized = eager
    fresh = FakeRun("fresh-id")
    if eager:
        target._checkpoint_record_wandb_run(fresh)
    checkpoint.restore_trainer_state(target, state, tmp_path)
    assert not target.wandb_initialized
    assert fresh.finished == eager
    kwargs = wandb_init_kwargs(target.wandb_config, {})
    assert kwargs["id"] == "saved-id"
    assert kwargs["resume"] == "allow"
    assert kwargs["project"] == "research"
    assert kwargs["entity"] == "team"


@pytest.mark.parametrize("override", [{"id": "caller-id"}, {"resume": "never"}])
def test_wandb_explicit_caller_choice_wins(tmp_path, override):
    source = TinyTrainer()
    source._checkpoint_record_wandb_run(FakeRun("saved-id"))
    target = TinyTrainer()
    target.wandb_config = dict(override)
    target.wandb_initialized = True
    run = FakeRun("caller-id")
    target._checkpoint_record_wandb_run(run)
    checkpoint.restore_trainer_state(
        target, checkpoint.capture_trainer_state(source), tmp_path
    )
    assert target.wandb_config == override
    assert target.wandb_initialized
    assert not run.finished


def test_wandb_default_kwargs_unchanged():
    assert wandb_init_kwargs({"project": "p"}, {}) == {
        "project": "p",
        "entity": None,
        "name": None,
        "config": {},
    }


def test_external_wandb_run_is_not_finished_or_partially_restored(tmp_path):
    source = TinyTrainer()
    source._checkpoint_record_wandb_run(FakeRun("saved"))
    target = TinyTrainer()
    target.wandb_config = {}
    target.wandb_initialized = True
    external = FakeRun("external")
    target._checkpoint_record_wandb_run(external, previous_run=external)
    before = checkpoint.capture_trainer_state(target)
    with pytest.raises(ValueError, match="external W&B"):
        checkpoint.restore_trainer_state(
            target, checkpoint.capture_trainer_state(source), tmp_path
        )
    assert not external.finished
    assert_nested_equal(checkpoint.capture_trainer_state(target), before)


def test_automatic_save_does_not_overwrite_foreign_run(tmp_path):
    first, second = TinyTrainer(), TinyTrainer()
    with first._checkpoint_context(checkpoint_dir=tmp_path):
        first._checkpoint_commit({"epoch": 1})
    saved = (tmp_path / "checkpoint-1" / checkpoint.STATE_FILENAME).read_bytes()
    with pytest.raises(FileExistsError, match="another run"):
        with second._checkpoint_context(
            checkpoint_dir=tmp_path, checkpoint_total_limit=1
        ):
            second._checkpoint_commit({"epoch": 1})
    assert (tmp_path / "checkpoint-1" / checkpoint.STATE_FILENAME).read_bytes() == saved


def test_automatic_save_expands_user_directory(tmp_path, monkeypatch):
    original = Path.expanduser
    monkeypatch.setattr(
        Path,
        "expanduser",
        lambda self: tmp_path / "runs" if str(self) == "~/runs" else original(self),
    )
    trainer = TinyTrainer()
    with trainer._checkpoint_context(checkpoint_dir="~/runs", checkpoint_total_limit=1):
        trainer._checkpoint_commit({"epoch": 1})
        trainer._checkpoint_commit({"epoch": 2})
    assert (tmp_path / "runs" / "checkpoint-2" / checkpoint.STATE_FILENAME).exists()
    assert not (tmp_path / "runs" / "checkpoint-1").exists()


def test_cuda_rng_remap_supports_unused_device_count_change(monkeypatch):
    trainer = object()
    state = {
        "rng": {"cuda": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]},
        "rng_devices": {"agents.0": "cuda:2", "critics.0": "cuda:0"},
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        checkpoint,
        "_rng_model_devices",
        lambda trainer: {"agents.0": "cuda:1", "critics.0": "cuda:0"},
    )
    assert checkpoint._cuda_rng_routes(trainer, state) == {1: 2, 0: 0}


@pytest.mark.parametrize(
    "saved,current",
    [
        (
            {"agents.0": "cuda:0", "agents.1": "cuda:1"},
            {"agents.0": "cuda:0", "agents.1": "cuda:0"},
        ),
        (
            {"agents.0": "cuda:0", "agents.1": "cuda:0"},
            {"agents.0": "cuda:0", "agents.1": "cuda:1"},
        ),
    ],
)
def test_ambiguous_actor_rng_merge_or_split_is_rejected(monkeypatch, saved, current):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(checkpoint, "_rng_model_devices", lambda trainer: current)
    with pytest.raises(ValueError, match="merge or split"):
        checkpoint._cuda_rng_routes(
            object(),
            {
                "rng": {"cuda": [torch.tensor([1]), torch.tensor([2])]},
                "rng_devices": saved,
            },
        )


@pytest.mark.parametrize("source,target", [("cpu", "cuda:0"), ("cuda:0", "cpu")])
def test_model_rng_backend_changes_rejected_without_cuda_runtime(
    monkeypatch, source, target
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        checkpoint, "_rng_model_devices", lambda trainer: {"agents.0": target}
    )
    with pytest.raises(ValueError, match="device backends"):
        checkpoint._cuda_rng_routes(
            object(), {"rng": {}, "rng_devices": {"agents.0": source}}
        )


def test_lazy_reward_rng_device_is_resolved_before_model_load(monkeypatch):
    trainer = SimpleNamespace(reward_device=torch.device("cuda:1"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        checkpoint, "_rng_model_devices", lambda trainer: {"agents.0": "cuda:0"}
    )
    state = {
        "rng": {"cuda": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]},
        "rng_devices": {"agents.0": "cuda:0", "reward_model.0": "cuda:2"},
    }
    assert checkpoint._cuda_rng_routes(trainer, state) == {0: 0, 1: 2}


def test_lazy_reward_implicit_cuda_device_uses_current_index(monkeypatch):
    trainer = SimpleNamespace(reward_device=torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        checkpoint, "_rng_model_devices", lambda trainer: {"agents.0": "cuda:0"}
    )
    state = {
        "rng": {"cuda": [torch.tensor([1]), torch.tensor([2])]},
        "rng_devices": {"agents.0": "cuda:0", "reward_model.0": "cuda:1"},
    }
    assert checkpoint._cuda_rng_routes(trainer, state) == {0: 0, 1: 1}


@pytest.mark.parametrize("probe_device", ["cpu", "meta"])
def test_rng_probe_override_uses_planned_placement(probe_device, monkeypatch):
    trainer = SimpleNamespace(
        agents=[torch.nn.Linear(1, 1, device=probe_device)],
        _checkpoint_rng_devices_override={"agents.0": "cuda:2"},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    state = {
        "rng": {"cuda": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]},
        "rng_devices": {"agents.0": "cuda:1"},
    }
    assert checkpoint._rng_model_devices(trainer) == {"agents.0": "cuda:2"}
    assert checkpoint._cuda_rng_routes(trainer, state)[2] == 1


def test_nested_rng_routes_restore_latest_parent_rng_values(monkeypatch):
    trainer = SimpleNamespace(_checkpoint_rng_devices_override={"agents.0": "cuda:0"})
    state = {
        "rng": {"cuda": [torch.tensor([11]), torch.tensor([22]), torch.tensor([33])]},
        "rng_devices": {"agents.0": "cuda:0"},
    }
    nested = [
        {
            "saved": {"agents.0": "cuda:0", "critics.0": "cuda:1"},
            "current": {"agents.0": "cuda:0", "critics.0": "cuda:2"},
        }
    ]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state_all", lambda: [torch.tensor([99])] * 3
    )
    restored = []
    monkeypatch.setattr(checkpoint, "restore_rng_state", restored.append)
    checkpoint._restore_trainer_rng(trainer, state, nested)
    assert [value.item() for value in restored[0]["cuda"]] == [11, 22, 22]


def test_nested_rng_routes_cannot_conflict_with_parent_streams(monkeypatch):
    trainer = SimpleNamespace(_checkpoint_rng_devices_override={"agents.0": "cuda:0"})
    state = {
        "rng": {"cuda": [torch.tensor([11]), torch.tensor([22])]},
        "rng_devices": {"agents.0": "cuda:0"},
    }
    nested = [{"saved": {"critics.0": "cuda:1"}, "current": {"critics.0": "cuda:0"}}]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="merge or split"):
        checkpoint._cuda_rng_routes(trainer, state, nested)


def test_lazy_factory_device_mismatch_rolls_back_before_state_load(monkeypatch):
    state = checkpoint.capture_trainer_state(TinyTrainer())
    target = TinyTrainer()
    target.reward_model = target.reward_optimizer = None
    target.reward_device = torch.device("cpu")
    before = {
        name: value.clone() for name, value in target.agents[0].state_dict().items()
    }
    rng = torch.get_rng_state().clone()

    def initialize():
        target.reward_model = torch.nn.Linear(3, 1)
        target.reward_optimizer = torch.optim.AdamW(target.reward_model.parameters())

    actual_devices = checkpoint._rng_model_devices

    def devices(trainer):
        result = actual_devices(trainer)
        if trainer is target and trainer.reward_model is not None:
            result["reward_model.0"] = "cuda:0"
        return result

    target._init_reward_model = initialize
    monkeypatch.setattr(checkpoint, "_rng_model_devices", devices)
    with pytest.raises(ValueError, match="device backends"):
        checkpoint.restore_trainer_state(target, state)
    assert target.reward_model is None
    assert target.reward_optimizer is None
    assert_nested_equal(target.agents[0].state_dict(), before)
    assert torch.equal(torch.get_rng_state(), rng)


def _attach_rng_consuming_logger(trainer, *, initialized):
    import random
    import numpy as np

    trainer.wandb_config = {}
    trainer.wandb_initialized = False
    trainer.logging_initializations = 0

    def initialize():
        random.random()
        np.random.random()
        torch.rand(())
        trainer.logging_initializations += 1
        trainer.wandb_initialized = True

    trainer._init_wandb = initialize
    if initialized:
        initialize()  # Match ordinary eager constructor logging.
    return trainer


@pytest.mark.parametrize("kind", ["magrpo", "actor_critic"])
def test_rl_resume_logging_consumes_rng_without_changing_training(tmp_path, kind):
    from test_checkpoint_rl import (
        _Interrupted,
        _TinyMAGRPO,
        _TinyActorCritic,
        _make,
        _run,
        _rng_state,
        _assert_rng_equal,
    )

    trainer_class = _TinyMAGRPO if kind == "magrpo" else _TinyActorCritic
    baseline = _attach_rng_consuming_logger(_make(trainer_class), initialized=True)
    _run(baseline)
    expected_rng = _rng_state()

    interrupted = _attach_rng_consuming_logger(
        _make(trainer_class, 1), initialized=True
    )
    with pytest.raises(_Interrupted):
        _run(interrupted, checkpoint_dir=tmp_path)
    resumed = _attach_rng_consuming_logger(_make(trainer_class), initialized=False)
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
    _assert_rng_equal(_rng_state(), expected_rng)


@pytest.mark.parametrize("kind", ["madpo", "iterative", "marlhf"])
def test_preference_resume_logging_consumes_rng_without_changing_training(
    tmp_path, kind, monkeypatch
):
    from test_checkpoint_preference import toy_trainer, iterative_trainer
    from test_checkpoint_rl import _rng_state, _assert_rng_equal
    import wandb

    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)

    def make(label, initialized):
        seed(183)
        if kind == "madpo":
            trainer = toy_trainer()
        elif kind == "iterative":
            trainer = iterative_trainer(tmp_path / label)
        else:
            trainer = marlhf_trainer()
        return _attach_rng_consuming_logger(trainer, initialized=initialized)

    baseline = make("baseline", True)
    baseline.train()
    expected_rng = _rng_state()
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
    _assert_rng_equal(_rng_state(), expected_rng)
