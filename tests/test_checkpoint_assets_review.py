"""Regression coverage for portable preference assets and generation boundaries."""

import errno
import json
import os
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import pytest
import torch

from comlrl.trainers.preference.checkpointing import (
    load_preference_assets,
    prune_preference_asset_pool,
    save_preference_assets,
    validate_preference_assets,
)
from comlrl.trainers.preference.iterative import PreferenceReplayShard
from test_checkpoint_preference import (
    Interrupted,
    assert_same_training,
    iterative_trainer,
    pair,
    seed,
    stop_after,
    toy_trainer,
)


def asset_trainer(directory):
    return SimpleNamespace(
        args=SimpleNamespace(
            preference_replay_dir=str(directory / "replay"),
            policy_checkpoint_dir=str(directory / "policies"),
        ),
        wandb_config={"output_dir": str(directory)},
        agents=[object()],
        _preference_replay_shards=lambda: [],
        _preference_replay_shards_state=[],
    )


def source_assets(directory):
    trainer = asset_trainer(directory)
    replay = directory / "replay" / "iteration_0001.json"
    replay.parent.mkdir(parents=True)
    replay.write_text(json.dumps({"iteration": 1, "num_pairs": 1, "pairs": [{}]}))
    trainer._preference_replay_shards_state = [PreferenceReplayShard(1, str(replay), 1)]
    history = directory / "policies"
    weights = history / "initial" / "agent_0" / "weights.bin"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"unchanged historical policy")
    trainer._policy_checkpoint_dir_path = str(history)
    return trainer, replay, weights


def test_assets_restore_to_new_run_with_independent_writable_files(tmp_path):
    source, _, _ = source_assets(tmp_path / "original")
    checkpoint = tmp_path / "checkpoints" / "checkpoint-1"
    manifest = save_preference_assets(source, checkpoint)
    portable = tmp_path / "portable"
    shutil.copytree(checkpoint, portable)
    shutil.rmtree(tmp_path / "original")
    shutil.rmtree(tmp_path / "checkpoints")

    target = asset_trainer(tmp_path / "new-run")
    validate_preference_assets(target, portable, manifest)
    load_preference_assets(target, portable, manifest)
    assert Path(target._preference_replay_dir_path) == tmp_path / "new-run" / "replay"
    assert Path(target._policy_checkpoint_dir_path) == tmp_path / "new-run" / "policies"
    assert (
        Path(target._reward_distribution_dir_path)
        == tmp_path / "new-run" / "reward_distributions"
    )
    restored = Path(target._policy_checkpoint_dir_path) / "initial/agent_0/weights.bin"
    saved = portable / "assets/policy_history/initial/agent_0/weights.bin"
    assert restored.stat().st_ino != saved.stat().st_ino
    assert restored.stat().st_mode & 0o200
    restored.write_bytes(b"new run changed its copy")
    assert saved.read_bytes() == b"unchanged historical policy"
    restored_replay = Path(target._preference_replay_shards_state[0].path)
    restored_replay.write_text("new run changed its replay")
    assert (
        json.loads((portable / manifest["replay"][0]["path"]).read_text())["num_pairs"]
        == 1
    )


def test_existing_run_outputs_are_preserved_in_persistent_resume_subdirectories(
    tmp_path,
):
    source, _, _ = source_assets(tmp_path / "original")
    checkpoint = tmp_path / "checkpoint"
    manifest = save_preference_assets(source, checkpoint)
    target = asset_trainer(tmp_path / "new-run")
    for name in ("replay", "policies", "reward_distributions"):
        directory = tmp_path / "new-run" / name
        directory.mkdir(parents=True)
        (directory / "keep.txt").write_text("existing output")
    load_preference_assets(target, checkpoint, manifest)
    for attribute, name in (
        ("_preference_replay_dir_path", "replay"),
        ("_policy_checkpoint_dir_path", "policies"),
        ("_reward_distribution_dir_path", "reward_distributions"),
    ):
        destination = Path(getattr(target, attribute))
        assert destination.parent == tmp_path / "new-run" / name
        assert destination.name.startswith("resume-")
        assert (destination.parent / "keep.txt").read_text() == "existing output"


def test_resume_rejects_output_inside_source_checkpoint_before_creating_it(tmp_path):
    source, _, _ = source_assets(tmp_path / "original")
    checkpoint = tmp_path / "checkpoint"
    manifest = save_preference_assets(source, checkpoint)
    target = asset_trainer(tmp_path / "new-run")
    target.args.preference_replay_dir = str(checkpoint / "forbidden-output")
    with pytest.raises(ValueError, match="outside the source checkpoint"):
        load_preference_assets(target, checkpoint, manifest)
    assert not (checkpoint / "forbidden-output").exists()
    assert not (tmp_path / "new-run").exists()


def test_unsafe_restore_destination_rejected_before_model_mutation(tmp_path):
    seed(10)
    source = iterative_trainer(tmp_path / "original")
    source._update_from_preference_batch([pair(4)])
    source.env_step = 99
    checkpoint = tmp_path / "checkpoint"
    source.save_checkpoint(checkpoint)
    seed(21)
    target = iterative_trainer(tmp_path / "new-run")
    target.args.policy_checkpoint_dir = str(checkpoint / "forbidden-output")
    before = target.agents[0].weight.detach().clone()
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="outside the source checkpoint"):
        target.load_checkpoint(checkpoint)
    assert torch.equal(target.agents[0].weight, before)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert not target.optimizers[0].state
    assert target.env_step == 0
    assert not hasattr(target, "_checkpoint_progress")
    assert not (checkpoint / "forbidden-output").exists()
    assert not (tmp_path / "new-run").exists()


@pytest.mark.parametrize(
    "config", ["checkpoint", "wandb", "sections", "explicit-parent"]
)
def test_resume_fallback_stays_in_requested_run(tmp_path, config):
    source, _, _ = source_assets(tmp_path / "original")
    checkpoint = tmp_path / "source-checkpoint"
    manifest = save_preference_assets(source, checkpoint)
    run = tmp_path / "new-run"
    target = asset_trainer(run)
    target.args.preference_replay_dir = None
    target.args.policy_checkpoint_dir = None
    target.wandb_config = None
    if config == "checkpoint":
        target._checkpoint_directory = str(run)
    elif config == "wandb":
        target.wandb_config = {"output_dir": str(run)}
    elif config == "sections":
        target.wandb_config = {"config_sections": {"output": {"base_dir": str(run)}}}
    else:
        target.args.policy_checkpoint_dir = str(run / "policies")
    load_preference_assets(target, checkpoint, manifest)
    for attribute in (
        "_preference_replay_dir_path",
        "_policy_checkpoint_dir_path",
        "_reward_distribution_dir_path",
    ):
        assert Path(getattr(target, attribute)).is_relative_to(run)


@pytest.mark.parametrize("config", ["checkpoint", "wandb", "explicit"])
def test_resumed_output_paths_match_source_tilde_conventions(
    tmp_path, monkeypatch, config
):
    source, _, _ = source_assets(tmp_path / "original")
    checkpoint = tmp_path / "source-checkpoint"
    manifest = save_preference_assets(source, checkpoint)

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    target = asset_trainer(tmp_path / "unused")
    target.args.preference_replay_dir = None
    target.args.policy_checkpoint_dir = None
    target.wandb_config = None
    if config == "checkpoint":
        target._checkpoint_directory = "~/runs"
    elif config == "wandb":
        target.wandb_config = {"output_dir": "~/runs"}
    else:
        target.args.preference_replay_dir = "~/runs/replay"
        target.args.policy_checkpoint_dir = "~/runs/policies"
    load_preference_assets(target, checkpoint, manifest)
    expected = tmp_path / ("home/runs" if config == "checkpoint" else "~/runs")
    for attribute in (
        "_preference_replay_dir_path",
        "_policy_checkpoint_dir_path",
        "_reward_distribution_dir_path",
    ):
        assert Path(getattr(target, attribute)).is_relative_to(expected)
    assert (tmp_path / "~").exists() == (config != "checkpoint")


def test_snapshots_share_immutable_assets_without_linking_live_history(tmp_path):
    source, replay, weights = source_assets(tmp_path / "run")
    root = tmp_path / "checkpoints"
    first, second, third = [root / f"checkpoint-{index}" for index in (1, 2, 3)]
    first_manifest = save_preference_assets(source, first)
    save_preference_assets(source, second)
    relative = "assets/policy_history/initial/agent_0/weights.bin"
    saved = first / relative
    assert saved.stat().st_ino == (second / relative).stat().st_ino
    assert saved.stat().st_ino != weights.stat().st_ino
    assert not saved.stat().st_mode & 0o222
    assert weights.stat().st_mode & 0o200
    assert (
        first / first_manifest["replay"][0]["path"]
    ).stat().st_ino != replay.stat().st_ino
    weights.write_bytes(b"a later policy snapshot")
    save_preference_assets(source, third)
    assert saved.read_bytes() == b"unchanged historical policy"
    assert (third / relative).read_bytes() == b"a later policy snapshot"
    assert (third / relative).stat().st_ino != saved.stat().st_ino
    # Every directory is portable even when the shared pool is removed.
    shutil.rmtree(root / ".comlrl-preference-assets")
    validate_preference_assets(source, first, first_manifest)


def test_asset_pool_pruning_keeps_referenced_blobs_and_skips_active_staging(tmp_path):
    source, _, _ = source_assets(tmp_path / "run")
    root = tmp_path / "checkpoints"
    checkpoint = root / "checkpoint-1"
    save_preference_assets(source, checkpoint)
    pool = root / ".comlrl-preference-assets"
    original = set(pool.iterdir())
    prune_preference_asset_pool(root)
    assert set(pool.iterdir()) == original
    shutil.rmtree(checkpoint)
    staging = root / ".checkpoint-2-pending" / "assets"
    staging.mkdir(parents=True)
    prune_preference_asset_pool(root)
    assert set(pool.iterdir()) == original
    shutil.rmtree(staging.parent)
    prune_preference_asset_pool(root)
    assert not list(pool.iterdir())


def test_snapshot_copy_fallback_when_filesystem_has_no_hardlinks(tmp_path, monkeypatch):
    source, _, weights = source_assets(tmp_path / "run")

    def unsupported_link(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "hardlinks unsupported")

    monkeypatch.setattr(os, "link", unsupported_link)
    checkpoint = tmp_path / "checkpoint"
    manifest = save_preference_assets(source, checkpoint)
    validate_preference_assets(source, checkpoint, manifest)
    saved = checkpoint / "assets/policy_history/initial/agent_0/weights.bin"
    assert saved.read_bytes() == weights.read_bytes()
    assert saved.stat().st_ino != weights.stat().st_ino


def test_madpo_commits_generated_pairs_before_first_update_and_large_interval(tmp_path):
    seed(12)
    baseline = toy_trainer()
    baseline.train()
    expected_random = (random.random(), torch.rand(()).item())
    seed(12)
    interrupted = toy_trainer()
    stop_after(interrupted, lambda progress: progress["epoch"] == 0)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path, checkpoint_interval=100)
    assert not interrupted.optimizers[0].state
    assert interrupted.events == []
    assert (tmp_path / "checkpoint-1" / "training_state.pt").is_file()

    seed(999)
    resumed = toy_trainer()
    resumed._build_preference_dataset = lambda **kwargs: pytest.fail(
        "regenerated pairs"
    )
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoint-1")
    assert_same_training(baseline, resumed)
    assert baseline.events == resumed.events
    assert expected_random == (random.random(), torch.rand(()).item())


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
